import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torch.nn.functional as F
import time
import copy
import argparse
import statistics
from efficientnet_pytorch import EfficientNet

batch_size = 8

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        return out

class DisentanglingRepresentationAdaptationNetwork(nn.Module):
    def __init__(self, num_classes=1000, use_dropout=False):
        super(DisentanglingRepresentationAdaptationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1, use_dropout=use_dropout)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2, use_dropout=use_dropout)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2, use_dropout=use_dropout)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2, use_dropout=use_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride, use_dropout):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_dropout=use_dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def main(args):
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print("Using CUDA")
    device = torch.device("cuda:0" if use_gpu else "cpu")

    epochs = args.epochs
    source_dataset = args.source
    target_dataset = args.target
    enable_transfer = args.transferlearning

    if source_dataset == target_dataset:
        print("Same source and target dataset. Exiting!")
        exit()

    if source_dataset == 'BOE':
        s_data_dir = 'BOE_split_by_person'
    elif source_dataset == 'CELL':
        s_data_dir = './OCT2017'
    elif source_dataset =='TMI':
        s_data_dir = './TMIdata_split_by_person'

    if target_dataset == 'BOE':
        t_data_dir = './BOE_split_by_person'
    elif target_dataset == 'CELL':
        t_data_dir = './OCT2017'
    elif target_dataset =='TMI':
        t_data_dir = 'TMIdata_split_by_person'

    print(f"Loading {source_dataset} dataset as Source from {s_data_dir}")
    print(f"Loading {target_dataset} dataset as Target from {t_data_dir}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    source_dataset = datasets.ImageFolder(os.path.join(s_data_dir), data_transforms['train'])
    target_dataset = datasets.ImageFolder(os.path.join(t_data_dir), data_transforms['train'])

    train_size = int(0.8 * len(source_dataset))
    val_size = len(source_dataset) - train_size
    source_train, source_val = random_split(source_dataset, [train_size, val_size])

    train_size = int(0.8 * len(target_dataset))
    val_size = len(target_dataset) - train_size
    target_train, target_val = random_split(target_dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(target_val, batch_size=batch_size, shuffle=False, num_workers=4)
    }

    model = EfficientNet.from_pretrained('efficientnet-b0')

    if enable_transfer:
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, len(source_dataset.classes))
    else:
        model._fc = nn.Linear(model._fc.in_features, len(target_dataset.classes))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, val_acc_history, train_acc_history = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)
    
    source_dataset_name = os.path.basename(source_dataset.root)
    target_dataset_name = os.path.basename(target_dataset.root)

    torch.save(model.state_dict(), f'./{source_dataset_name}_to_{target_dataset_name}_model.pth')

    # Convert tensors to plain floats
    train_acc_history = [acc.item() for acc in train_acc_history]
    val_acc_history = [acc.item() for acc in val_acc_history]

    # Print the accuracies
    print("Training accuracy history:", train_acc_history)
    print("Validation accuracy history:", val_acc_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source',
                        help='source dataset, choose from [BOE, CELL, TMI]',
                        type=str,
                        choices=['BOE', 'CELL', 'TMI'],
                        default='TMI')

    parser.add_argument('-t', '--target',
                        help='target dataset, choose from [BOE, CELL, TMI]',
                        type=str,
                        choices=['BOE', 'CELL', 'TMI'],
                        default='CELL')

    parser.add_argument('-e', '--epochs',
                        help='training epochs',
                        type=int,
                        default=30)

    parser.add_argument('-l', '--transferlearning',
                        help='Set transfer learning or not',
                        action='store_true')

    args = parser.parse_args()
    main(args)

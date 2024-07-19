import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

# Define batch size
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
    def __init__(self, num_classes=1000):
        super(DisentanglingRepresentationAdaptationNetwork, self).__init__()
        # Define your convolutional layers here
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Define other layers...
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Pass through other layers...
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=1000):
        super(EfficientNetEncoder, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        
        # Replace the final fully connected layer
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove the original fully connected layer
        
        # Define a new fully connected layer
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.efficientnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=64, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )
        self.w = nn.ParameterDict()
        w, h = imsize
        for cv in converts:
            self.w[cv] = nn.Parameter(torch.ones(1, ch, h//down_scale, w//down_scale), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles = dict(), dict()
        for key in features.keys():
            styles[key] = self.conv(features[key])  # equals to F - wS(F) see eq.(2)
            contents[key] = features[key] - styles[key]  # equals to wS(F)
            if '2' in key:  # for 3 datasets: source-mid-target
                source, target = key.split('2')
                contents[target] = contents[key]

        if converts is not None:  # separate features of converted images to compute consistency loss.
            for cv in converts:
                source, target = cv.split('2')
                contents[cv] = self.w[cv] * contents[source]
        return contents, styles

class Generator(nn.Module):
    def __init__(self, channels=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, content, style):
        x = content + style
        print("Generator - Shape before view:", x.shape)  # Print the shape
        # Comment out the view operation for now to prevent the error
        # x = x.view(-1, 3, 1, 1)  
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=input_dims, out_channels=hidden_dims, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims, out_channels=hidden_dims * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims * 2, out_channels=hidden_dims * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims * 4, out_channels=output_dims, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out

def entropy_loss(p_logit, temperature=1.0, label_smoothing=0.0):
    p = F.softmax(p_logit / temperature, dim=-1)
    
    if label_smoothing > 0.0:
        num_classes = p_logit.size(-1)
        smooth = label_smoothing / num_classes
        p = (1 - label_smoothing) * p + smooth
    
    return torch.mean(torch.sum(p * F.log_softmax(p_logit / temperature, dim=-1), dim=-1))

def test(encoder, classifier, dataloader_test, dataset_size_test):
    since = time.time()
    acc_test = 0
    
    for inputs, labels in dataloader_test:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = classifier(encoder(inputs))
            _, preds = torch.max(outputs, 1)
            acc_test += torch.sum(preds == labels.data)
    
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Accuracy on test set: {acc_test.double() / dataset_size_test:.4f}')
    
    return acc_test.double() / dataset_size_test

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        #print("Classifier - Shape before fully connected layer:", x.shape)  # Debugging print statement
        x = self.fc(x)
        #print("Classifier - Shape after fully connected layer:", x.shape)  # Debugging print statement
        return x
    
def main(args):
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print("Using CUDA")

    epochs = args.epochs
    source_dataset = args.source
    target_dataset = args.target
    enable_transfer = args.transferlearning

    if source_dataset == target_dataset:
        print("Same source and target dataset. Exiting!")
        exit()

    if source_dataset == 'BOE':
        print("Loading BOE data set as Source")
        s_data_dir = 'BOE_split_by_person'
    elif source_dataset == 'CELL':
        print("Loading CELL data set as Source")
        s_data_dir = './OCT2017'
    elif source_dataset == 'TMI':
        print("Loading TMI data set as Source")
        s_data_dir = './TMIdata_split_by_person'

    if target_dataset == 'BOE':
        print("Loading BOE data set as Target")
        t_data_dir = './BOE_split_by_person'
    elif target_dataset == 'CELL':
        print("Loading CELL data set as Target")
        t_data_dir = './OCT2017'
    elif target_dataset == 'TMI':
        print("Loading TMI data set as Target")
        t_data_dir = 'TMIdata_split_by_person'

    # Define data transformations
    # Modify the data transforms to ensure the input tensor has the correct shape
    # Define data transformations with normalization and resizing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization
        ]),
    }

    # Create data loaders for both the source and target datasets
    source_dataset = datasets.ImageFolder(os.path.join(s_data_dir, 'train'), data_transforms['train'])
    target_dataset = datasets.ImageFolder(os.path.join(t_data_dir, 'train'), data_transforms['train'])

    # Split the source dataset into training and testing sets
    train_size = int(0.8 * len(source_dataset))  # 80% for training, 20% for testing
    test_size = len(source_dataset) - train_size
    train_dataset, test_dataset = random_split(source_dataset, [train_size, test_size])

    # Split the target dataset into training and testing sets
    train_target_size = int(0.8 * len(target_dataset))
    test_target_size = len(target_dataset) - train_target_size
    train_target_dataset, test_target_dataset = random_split(target_dataset, [train_target_size, test_target_size])

    # Create data loaders for both training and testing sets for source and target datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_target_dataloader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_target_dataloader = DataLoader(test_target_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the number of classes
    num_classes = len(target_dataset.classes)

    # Load encoder and classifier
    if args.transferlearning:
        encoder = EfficientNetEncoder(model_name='efficientnet-b0', num_classes=1000).to(device)
        classifier = Classifier(num_classes=num_classes).to(device)
    else:
        encoder = DisentanglingRepresentationAdaptationNetwork(num_classes=1000).to(device)
        classifier = Classifier(num_classes=num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        encoder.train()
        classifier.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader if args.transferlearning else train_target_dataloader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            features = encoder(inputs)
            #print("Shape of features:", features.shape)  # Print the shape
            outputs = classifier(features)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            # Print batch statistics every few batches
            if i % 100 == 0:  # Adjust the frequency based on your preference
                batch_loss = running_loss / (batch_size * i)
                batch_acc = running_corrects.double() / (batch_size * i)
                print(f'Batch {i}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}')

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_dataloader.dataset) if args.transferlearning else running_loss / len(train_target_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset) if args.transferlearning else running_corrects.double() / len(train_target_dataloader.dataset)
        domain = "Source" if args.transferlearning else "Target"
        print(f'{domain} Domain Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        scheduler.step()

    print('Training completed')

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

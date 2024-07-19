import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import time
import copy
import argparse
import statistics
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
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

class Encoder(nn.Module):
    def __init__(self, channels=3):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        output = self.model(x)
        return output



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
        return self.model(content+style)
    
class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=input_dims, out_channels=hidden_dims, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims, out_channels=hidden_dims*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims*2, out_channels=hidden_dims*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=hidden_dims*4, out_channels=output_dims, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out

def entropy_loss(p_logit, temperature=1.0, label_smoothing=0.0):
    p = F.softmax(p_logit / temperature, dim=-1)
    
    if label_smoothing > 0.0:
        num_classes = p_logit.size(-1)
        smoothed_labels = (1.0 - label_smoothing) * p + label_smoothing / num_classes
        return -torch.sum(smoothed_labels * F.log_softmax(p_logit / temperature, dim=-1)) / p_logit.size(0)
    else:
        return -torch.sum(p * F.log_softmax(p_logit / temperature, dim=-1)) / p_logit.size(0)


# Define test function
def test(encoder, classifier, dataloader_test, dataset_size_test):
    since = time.time()
    acc_test = 0
    
    for inputs, labels in dataloader_test:
        encoder.eval()
        classifier.eval()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                
            inputs, labels = Variable(inputs), Variable(labels)
            features = encoder(inputs)
            outputs = classifier(features.view(features.size(0), -1))
            _, preds = torch.max(outputs.data, 1)
            acc_test += torch.sum(preds == labels.data).item()

    elapsed_time = time.time() - since
    print("Test completed in {:.2f}s".format(elapsed_time))
    avg_acc = float(acc_test) / dataset_size_test
    print("Test accuracy={:.4f}".format(avg_acc))
    print()
    
    return avg_acc

def train_src(encoder, classifier, dataloader_train, dataloader_val, epochs, save_name):
    since = time.time()
    
    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001, momentum=0.9)
    
    best_encoder = copy.deepcopy(encoder.state_dict())
    best_classifier = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloader_train:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                features = encoder(inputs)
                outputs = classifier(features.view(features.size(0), -1))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader_train.dataset)
        epoch_acc = running_corrects.double() / len(dataloader_train.dataset)
        
        print('Epoch {}/{} - Train Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epochs, epoch_loss, epoch_acc))
        
        val_acc = test(encoder, classifier, dataloader_val, len(dataloader_val.dataset))
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_classifier = copy.deepcopy(classifier.state_dict())
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    
    try:
        torch.save(encoder.state_dict(), save_name + 'DRANet_encoder.pt')
        torch.save(classifier.state_dict(), save_name + 'DRANet_classifier.pt')
        print("Model weights saved successfully.")
    except Exception as e:
        print("Error occurred while saving model weights:", e)
    
    return encoder, classifier

def train_tgt(src_encoder, src_classifier, tgt_encoder, netD, src_data_loader, tgt_data_loader, save_name, num_epochs=10):
    since = time.time()
    
    # Set train state for Dropout and BatchNorm layers
    src_encoder.eval()
    tgt_encoder.train()
    netD.train()
    
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()  # Assuming you use Mean Squared Error for reconstruction loss
    adversarial_loss = nn.BCELoss()  # Binary Cross Entropy Loss for adversarial loss
    
    optimizer_tgt = optim.SGD(tgt_encoder.parameters(), lr=0.001, momentum=0.9)
    optimizer_critic = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)
    
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))
    
    for epoch in range(num_epochs):
        data_zip = zip(src_data_loader, tgt_data_loader)
        
        for step, ((images_src, _), (images_tgt, _)) in enumerate(data_zip):
            if torch.cuda.is_available():
                images_src, images_tgt = images_src.cuda(), images_tgt.cuda()
                
            optimizer_critic.zero_grad()
            
            # Discriminator training
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat = feat_concat.view(feat_concat.size(0), -1)
            pred_concat = netD(feat_concat.detach())
            
            label_src = torch.ones(feat_src.size(0), dtype=torch.long)
            label_tgt = torch.zeros(feat_tgt.size(0), dtype=torch.long)
            label_concat = torch.cat((label_src, label_tgt), 0)
            
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()
            optimizer_critic.step()
            
            pred_cls = torch.argmax(pred_concat, dim=1)
            acc = torch.mean((pred_cls == label_concat).float())
            
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()
            
            # Target Encoder training
            feat_tgt = tgt_encoder(images_tgt)
            feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)
            outputs = src_classifier(feat_tgt)
            loss_em = entropy_loss(outputs)
            pred_tgt = netD(feat_tgt)
            
            label_tgt = torch.ones(feat_tgt.size(0), dtype=torch.long)
            loss_tgt = criterion(pred_tgt, label_tgt)
            
            # Reconstruction loss
            recon_loss = reconstruction_loss(tgt_encoder(images_tgt), images_tgt)
            
            # Adversarial loss
            ones = torch.ones_like(pred_tgt)
            adv_loss = adversarial_loss(pred_tgt, ones)
            
            # Total loss
            loss = loss_tgt + loss_em + recon_loss + adv_loss
            
            loss.backward()
            optimizer_tgt.step()
            
            if (step + 1) % 5 == 0:
                print("Epoch [{}/{}] Step [{}/{}]: d_loss={:.5f} | g_loss={:.5f} | EM_loss={:.5f} | Recon_loss={:.5f} | Adv_loss={:.5f} | acc={:.5f}"
                      .format(epoch + 1, num_epochs, step + 1, len_data_loader, loss_critic.item(), loss_tgt.item(), loss_em.item(), recon_loss.item(), adv_loss.item(), acc.item()))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    save_name = "C:\\Users\\admin\\Desktop\\Research - Minidu Wickramaarachchi\\OCT_DDA-main\\model_saved"
    if not os.path.exists(save_name):
        os.makedirs(save_name)

    torch.save(netD.state_dict(), os.path.join(save_name, "DRANet_netD.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(save_name, "DRANet_tgt_encoder.pt"))
    
    return tgt_encoder

#def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    model = model.to(device)  # Move model to appropriate device
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def loss_weights(task, dsets):
    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

        # MNIST <-> MNIST-M
        if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

        # MNIST <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
            alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
            alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

        # MNIST <-> MNIST-M <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

    elif task == 'seg':
        # GTA5 <-> Cityscapes
        alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
        alpha['style']['G2C'], alpha['style']['C2G'] = 5e3, 5e3
        alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
        alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[source], contents_converted[cv]) + \
                                F.l1_loss(styles[source], styles_converted[cv])
            return consistency_loss

    def forward(self, imgs, recon_imgs, real, fake, perceptual, perceptual_converted, style_gram, style_gram_converted, contents, styles, contents_converted, styles_converted, converts):
        return (
            self.recon(imgs, recon_imgs) +
            self.dis(real, fake) +
            self.gen(fake) +
            self.content_perceptual(perceptual, perceptual_converted) +
            self.style_perceptual(style_gram, style_gram_converted) +
            self.consistency(contents, styles, contents_converted, styles_converted, converts)
        )

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
        print(" Loading BOE data set as Source")
        s_data_dir = 'BOE_split_by_person'
        print(" Loading {} data set as Source".format(s_data_dir))
    elif source_dataset == 'CELL':
        print(" Loading CELL data set as Source")
        s_data_dir = './OCT2017'
    elif source_dataset =='TMI':
        print(" Loading TMI data set as Source ")
        s_data_dir = './TMIdata_split_by_person'

    if target_dataset == 'BOE':
        print(" Loading BOE data set as Target")
        t_data_dir = './BOE_split_by_person'
    elif target_dataset == 'CELL':
        print(" Loading CELL data set as Target")
        t_data_dir = './OCT2017'
    elif target_dataset =='TMI':
        print(" Loading TMI data set as Target ")
        t_data_dir = 'TMIdata_split_by_person'
        print(" Loading {} data set as Target ".format(t_data_dir))

    # Define data transformations
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
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_target_dataloader = torch.utils.data.DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_target_dataloader = torch.utils.data.DataLoader(test_target_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define your model
    if enable_transfer:
        model = DisentanglingRepresentationAdaptationNetwork().to(device)  # For source domain training
    else:
        model = DisentanglingRepresentationAdaptationNetwork().to(device)  # For target domain training

    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Initialize optimizer and learning rate scheduler
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    step_size = 5
    gamma = 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_dataloader if enable_transfer else train_target_dataloader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
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
        epoch_loss = running_loss / len(train_dataloader.dataset) if enable_transfer else running_loss / len(train_target_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset) if enable_transfer else running_corrects.double() / len(train_target_dataloader.dataset)
        domain = "Source" if enable_transfer else "Target"
        print(f'{domain} Domain Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


        print()

        # Optionally update learning rate
        scheduler.step()

    print('Training completed')

# Test the model
    #def test(model, test_loader, dataset_size):
        #model.eval()
        #running_corrects = 0

        #with torch.no_grad():
            #for inputs, labels in test_loader:
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                #outputs = model(inputs)
                #_, preds = torch.max(outputs, 1)
                #running_corrects += torch.sum(preds == labels.data)

        #accuracy = running_corrects.double() / dataset_size
        #print(f'Test Accuracy: {accuracy:.4f}')

    #print("Testing source model on source test data...")
    #test(model, test_dataloader, len(test_dataloader.dataset))

    #if enable_transfer:
        #print("Testing target model on target test data...")
        #test(model, test_target_dataloader, len(test_target_dataloader.dataset))

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

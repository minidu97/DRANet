import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import statistics
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchviz import make_dot
from torch.nn.utils import spectral_norm
from sklearn.metrics import confusion_matrix


batch_size = 8

def entropy_loss(p_logit):
    p = F.softmax(p_logit, dim=-1)
    return -1 * torch.sum(p * F.log_softmax(p_logit, dim=-1)) / p_logit.size()[0]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)

        out.add_(residual)
        out = self.relu(out)
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
        layers = [ResidualBlock(in_channels, out_channels, stride, use_dropout)]
        layers += [ResidualBlock(out_channels, out_channels, use_dropout=use_dropout) for _ in range(1, blocks)]
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
    def __init__(self, pretrained=True, source_domain='', target_domain=''):
        super(Encoder, self).__init__()
        self.pretrained = pretrained
        self.source_domain = source_domain
        self.target_domain = target_domain

        if self.pretrained:
            self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if self.source_domain and self.target_domain:
            self.separator = Separator(imsize=(224, 224), converts=[f'{source_domain}2{target_domain}'])

    def forward(self, x):
        if self.pretrained:
            features = self.encoder.extract_features(x)
        if self.source_domain and self.target_domain:
            contents, styles = self.separator(features)
            return contents, styles
        else:
            return features

class Separator(nn.Module):
    def __init__(self, imsize, converts, ch=64, down_scale=2):
        super(Separator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(inplace=True),
        )
        self.w = nn.ParameterDict()
        w, h = imsize
        for cv in converts:
            self.w[cv] = nn.Parameter(torch.ones(1, ch, h // down_scale, w // down_scale), requires_grad=True)

    def forward(self, features, converts=None):
        contents, styles = {}, {}
        for key in features:
            style = self.conv(features[key])
            content = features[key] - style
            styles[key] = style
            contents[key] = content
            if '2' in key:
                _, target = key.split('2')
                contents[target] = content

        if converts is not None:
            for cv in converts:
                source, target = cv.split('2')
                contents[cv] = self.w[cv] * contents[source]
        return contents, styles


class Generator(nn.Module):
    def __init__(self, channels=512):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            spectral_norm(nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, content, style):
        combined_input = content + style
        return self.model(combined_input)

class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            spectral_norm(nn.Linear(input_dims, hidden_dims)),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dims, hidden_dims)),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dims, hidden_dims)),
            nn.BatchNorm1d(hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Linear(hidden_dims, output_dims))
        )

    def forward(self, input):
        out = self.layer(input)
        return torch.sigmoid(out)

class ClassifierNetwork(nn.Module):
    def __init__(self, input_dims=1000, num_classes=3, dropout=0.5):
        super(ClassifierNetwork, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dims, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self.out_dims = num_classes

    def forward(self, x):
        x = self.fc_layers(x)
        return x
    
def test(encoder, classifier, dataloader_test, dataset_size_test):
    since = time.time()
    acc_test = 0
    for i, data in enumerate(dataloader_test):
        encoder.eval()
        classifier.eval()
        inputs, labels = data

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            features = encoder(inputs)
            outputs = classifier(features.view(features.size(0), -1))
            _, preds = torch.max(outputs.data, 1)
            acc_test += torch.sum(preds == labels.data).item()

        del inputs, labels, features, preds
        torch.cuda.empty_cache()

    elapsed_time = time.time() - since
    print("Test completed in {:.2f}s".format(elapsed_time))
    avg_acc = float(acc_test) / dataset_size_test
    print("test acc={:.4f}".format(avg_acc))
    print()
    torch.cuda.empty_cache()
    return avg_acc


def train_src(encoder, classifier, dataloader_train, dataloader_val, epochs, save_name, best_val_acc):
    since = time.time()
    """Train classifier for source domain."""
    lr = 0.001

    # Setup optimizer and initial criterion without weights
    optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_encoder = copy.deepcopy(encoder.state_dict())
    best_classifier = copy.deepcopy(classifier.state_dict())
    best_acc = 0.0

    # Determine the device from the encoder's parameters
    device = next(encoder.parameters()).device

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        running_loss_train = 0.0
        running_corrects_train = 0
        total_train = 0

        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients for optimizer
            optimizer.zero_grad()

            # Forward pass
            features = encoder(inputs)
            outputs = classifier(features.view(features.size(0), -1))
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        avg_loss_train = running_loss_train / total_train
        avg_acc_train = running_corrects_train / total_train

        encoder.eval()
        classifier.eval()

        running_loss_val = 0.0
        running_corrects_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader_val:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                features = encoder(inputs)
                outputs = classifier(features.view(features.size(0), -1))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss_val += loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss_val = running_loss_val / total_val
        avg_acc_val = running_corrects_val / total_val

        # Compute confusion matrix
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        #print(f"Confusion Matrix for epoch {epoch+1}:\n{cm}")

        # Adjust class weights based on confusion matrix
        class_counts = np.sum(cm, axis=1)
        class_weights = 1.0 / (class_counts / np.sum(class_counts))
        class_weights = torch.tensor(class_weights).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Update the learning rate
        scheduler.step()

        elapsed_time = time.time() - since
        print('Epoch {}/{} completed in {:.0f}m {:.0f}s'.format(epoch + 1, epochs, elapsed_time // 60, elapsed_time % 60))
        print("train loss={:.4f} train acc={:.4f}".format(avg_loss_train, avg_acc_train))
        print("val loss={:.4f} val acc={:.4f}".format(avg_loss_val, avg_acc_val))
        print()

        if avg_acc_val > best_val_acc:
            best_val_acc = avg_acc_val
            best_encoder = copy.deepcopy(encoder.state_dict())
            best_classifier = copy.deepcopy(classifier.state_dict())
            torch.save(encoder.state_dict(), save_name + '_best_encoder.pt')
            torch.save(classifier.state_dict(), save_name + '_best_classifier.pt')

        torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # Load best model weights
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)

    return encoder, classifier, best_val_acc



def train_tgt(src_encoder, src_classifier, tgt_encoder, netD, src_data_loader, tgt_data_loader, save_name, num_epochs=10, cycleGAN=False):
    since = time.time()
    ####################
    # 1. setup network #
    ####################

    # Set train state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.train()
    netD.train()

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer_tgt = optim.AdamW(tgt_encoder.parameters(), lr=0.0001)
    #optimizer_critic = optim.AdamW(netD.parameters(), lr=0.001)
    optimizer_tgt = optim.SGD(tgt_encoder.parameters(), lr=0.0001, momentum=0.9)
    optimizer_critic = optim.SGD(netD.parameters(), lr=0.001, momentum=0.9)

    # Learning rate scheduler
    scheduler_tgt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_tgt, mode='min', factor=0.1, patience=5)
    scheduler_critic = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_critic, mode='min', factor=0.1, patience=5)

    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    # Initialize variables to track best validation accuracy and corresponding model weights
    best_val_acc = 0.0
    best_tgt_encoder_weights = None

    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:

            ###########################
            # 2.1 train discriminator #
            ###########################

            # Make images variable
            images_src, images_tgt = images_src.cuda(), images_tgt.cuda()

            # Zero gradients for optimizer
            optimizer_critic.zero_grad()

            # Extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            feat_concat = feat_concat.view(feat_concat.size(0), -1)

            # Predict on discriminator
            pred_concat = netD(feat_concat.detach())

            # Prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long().cuda()
            label_tgt = torch.zeros(feat_tgt.size(0)).long().cuda()
            label_concat = torch.cat((label_src, label_tgt), 0)

            # Compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # Optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # Zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # Extract and target features
            feat_tgt = tgt_encoder(images_tgt)
            feat_tgt = feat_tgt.view(feat_tgt.size(0), -1)

            # Predict on Source classifier
            outputs = src_classifier(feat_tgt)
            # Calculate EM loss
            loss_em = entropy_loss(outputs)

            # Predict on discriminator
            pred_tgt = netD(feat_tgt)

            # Prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # Compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss = loss_tgt + loss_em
            loss.backward()

            # Optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if (step + 1) % 5 == 0:
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} | g_loss={:.5f} | EM_loss={:.5f} | acc={:.5f}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              loss_em.item(),
                              acc.item()))

        # Step the scheduler
        scheduler_tgt.step(loss_tgt)
        scheduler_critic.step(loss_critic)

        # Print epoch information
        print(f'Epoch {epoch + 1}/{num_epochs}')

    elapsed_time = time.time() - since
    print("Target Training completed in {:.2f}s".format(elapsed_time))

    # Saving best model weights
    torch.save(netD.state_dict(), save_name + "_netD.pt")
    torch.save(tgt_encoder.state_dict(), save_name + "_target_encoder_best.pt")

    return tgt_encoder

def main(args):
    # Ensure that your device is set to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_gpu = torch.cuda.is_available()
    if use_gpu: 
        print("Using CUDA")

    epochs = args.epochs
    source_dataset = args.source
    target_dataset = args.target
    enable_transfer = (args.transferlearning == 1)

    if source_dataset == target_dataset:
        print("Same source and target dataset. Exit!")
        exit()

    if source_dataset == 'BOE':
        print(" Loading BOE data set as Source")
        s_data_dir = 'BOE_split_by_person'
        print(" Loading {} data set as Source".format(s_data_dir))
    elif source_dataset == 'CELL':
        print(" Loading CELL data set as Source")
        s_data_dir = './OCT2017'
    elif source_dataset == 'TMI':
        print(" Loading TMI data set as Source ")
        s_data_dir = './TMIdata_split_by_person'

    if target_dataset == 'BOE':
        print(" Loading BOE data set as Target")
        t_data_dir = './BOE_split_by_person'
    elif target_dataset == 'CELL':
        print(" Loading CELL data set as Target")
        t_data_dir = './OCT2017'
    elif target_dataset == 'TMI':
        print(" Loading TMI data set as Target ")
        t_data_dir = 'TMIdata_split_by_person'
        print(" Loading {} data set as Target ".format(t_data_dir))

    iterations = 2

    batch_size = 18  # Define your batch size here

    TRAIN_S, VAL_S, TEST_S = 'train', 'val', 'test'
    TRAIN_T, TEST_T = 'train', 'test'

    if not os.path.exists('./DRANet_model_saved'):
        os.makedirs('./DRANet_model_saved')

    # Source Data Loading
    data_transform_s = {
        TRAIN_S: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        VAL_S: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        TEST_S: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

    image_dataset_s = {
        x: datasets.ImageFolder(os.path.join(s_data_dir, x), transform=data_transform_s[x])
        for x in [TRAIN_S, VAL_S, TEST_S]
    }

    dataloader_s = {
        x: torch.utils.data.DataLoader(image_dataset_s[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in [TRAIN_S, VAL_S, TEST_S]
    }

    dataset_sizes_src = {x: len(image_dataset_s[x]) for x in [TRAIN_S, VAL_S, TEST_S]}

    for x in [TRAIN_S, VAL_S, TEST_S]:
        print("Loaded {} images under Source {}".format(dataset_sizes_src[x], x))

    class_names = image_dataset_s[TRAIN_S].classes
    print("Classes: ", image_dataset_s[TRAIN_S].classes)

    # Target Data Loading
    data_transform_t = {
        TRAIN_T: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        TEST_T: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

    image_dataset_t = {
        x: datasets.ImageFolder(os.path.join(t_data_dir, x), transform=data_transform_t[x])
        for x in [TRAIN_T, TEST_T]
    }

    dataloader_t = {
        x: torch.utils.data.DataLoader(image_dataset_t[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in [TRAIN_T, TEST_T]
    }

    dataset_sizes_tgt = {x: len(image_dataset_t[x]) for x in [TRAIN_T, TEST_T]}

    for x in [TRAIN_T, TEST_T]:
        print("Loaded {} images under Target {}".format(dataset_sizes_tgt[x], x))

    class_names = image_dataset_t[TRAIN_T].classes
    print("Classes: ", image_dataset_t[TRAIN_T].classes)

    # Model Training
    test_acc = []
    test_acc_no_transfer = []
    saved_model_name = './result/' + source_dataset + '_to_' + target_dataset + '_best.pt'

    for iter in range(1, iterations + 1):

        save_name = './model_saved/DRANet' + source_dataset + '_to_' + target_dataset + '_iter' + str(iter)

        # Source Encoder and Classifier
        src_encoder = DisentanglingRepresentationAdaptationNetwork().to(device)
        

        src_classifier = ClassifierNetwork().to(device)
        #src_classifier = ClassifierNetwork(input_dims=512, num_classes=len(class_names)).to(device)  # Provide input_dims and num_classes

        # Discriminator
        netD = Discriminator(input_dims=512, hidden_dims=500, output_dims=2).to(device)

        if not enable_transfer:
            print("No Transfer Learning")

            source_encoder_name = save_name + '_source_encoder.pt'
            source_cls_name = save_name + '_source_classifier.pt'
            src_encoder.load_state_dict(torch.load(source_encoder_name, map_location=device))
            src_classifier.load_state_dict(torch.load(source_cls_name, map_location=device))

            print("Test src_encoder + src_classifier on Source Test dataset")
            src_acc = test(src_encoder, src_classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])
            test_acc_no_transfer.append(src_acc)

            print("Test src_encoder + src_classifier on Target Test dataset")
            tgt_acc = test(src_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])
            test_acc.append(tgt_acc)

            # Save predictions and labels
            save_name = './predictions/DRANet' + source_dataset + '_to_' + target_dataset + '_iter' + str(iter) + '_predictions.pt'

            os.makedirs(os.path.dirname(save_name), exist_ok=True)

            pred_result = {'y_true_src': src_acc[1], 'y_pred_src': src_acc[2], 'y_prob_src': src_acc[3],
                           'y_true_tgt': tgt_acc[1], 'y_pred_tgt': tgt_acc[2], 'y_prob_tgt': tgt_acc[3]}
            torch.save(pred_result, save_name)

        if enable_transfer:
            best_val_acc = 0.0  # Initialize the best validation accuracy variable

            print("DRANet Transfer Learning")
            src_encoder, src_classifier, best_val_acc = train_src(src_encoder, src_classifier, dataloader_s[TRAIN_S], dataloader_s[VAL_S], epochs, save_name, best_val_acc)

            source_encoder_name = save_name + '_best_encoder.pt'
            source_cls_name = save_name + '_best_classifier.pt'
            src_encoder.load_state_dict(torch.load(source_encoder_name, map_location=device))
            src_classifier.load_state_dict(torch.load(source_cls_name, map_location=device))

            for param in src_encoder.parameters():
                param.requires_grad = False
            for param in src_classifier.parameters():
                param.requires_grad = False

            print("Training encoder for the target domain...........................")

            tgt_encoder = DisentanglingRepresentationAdaptationNetwork().to(device)
            tgt_encoder.load_state_dict(src_encoder.state_dict())

            tgt_encoder = train_tgt(src_encoder, src_classifier, tgt_encoder, netD, dataloader_s[TRAIN_S],
                                    dataloader_t[TRAIN_T], save_name, epochs)
            print("Test src_encoder + src_classifier on Source Test dataset")
            test(src_encoder, src_classifier, dataloader_s[TEST_S], dataset_sizes_src[TEST_S])

            print("Test src_encoder + src_classifier on Target Test dataset")
            test(src_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])

            print("Test tgt_encoder + src_classifier on Target Test dataset")
            tgt_acc = test(tgt_encoder, src_classifier, dataloader_t[TEST_T], dataset_sizes_tgt[TEST_T])
            test_acc.append(tgt_acc)

    if enable_transfer:
        print('test_acc=', test_acc)
        test_acc_avg = sum(test_acc) / len(test_acc)
        test_acc_var = statistics.stdev(test_acc)
        print("Average test acc: %.4f" % test_acc_avg, '| Variance test: %.4f' % test_acc_var)
        print('Best validation accuracy:', best_val_acc)
    else:
        print("No transferring test_acc = ", test_acc_no_transfer)
        print('Best validation accuracy:', best_val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source',
                        help='source dataset, choose from [BOE,CELL,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='TMI')

    parser.add_argument('-t', '--target',
                        help='target dataset, choose from [BOE,CELL,TMI]',
                        type=str,
                        choices=['BOE','CELL','TMI'],
                        default='CELL')

    parser.add_argument('-e', '--epochs',
                        help='training epochs',
                        type=int,
                        default=30)

    parser.add_argument('-l', '--transferlearning',
                        help='Set transfer learning or not, 1=using transfer, 0=not using tranfer ',
                        type=int,
                        choices=[1,0],
                        default=1)

    args = parser.parse_args()
    main(args)

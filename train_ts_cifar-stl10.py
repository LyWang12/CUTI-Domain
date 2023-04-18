import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.getdata import Cus_Dataset
from utils.util import get_cifar_data, get_stl_data

SEED = 2023
lr = 0.0001
nepoch = 30
workers = 10
batch_size = 32
device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SOURCE = 'cifar'   # cifar, stl
TARGET = 'stl'   # cifar, stl

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


class CUTI(nn.Module):
    def __init__(self):
        super(CUTI, self).__init__()
    def forward(self, x):
        if self.training:
            batch_size, C = x.size()[0]//2, x.size()[1]
            style_mean, style_std = calc_ins_mean_std(x[:batch_size])
            conv_mean = nn.Conv2d(C, C, 1, bias=False).cuda()
            conv_std = nn.Conv2d(C, C, 1, bias=False).cuda()
            mean = torch.sigmoid(conv_mean(style_mean))
            std = torch.sigmoid(conv_std(style_std))
            x_a = x[batch_size:]*std+mean
            x = torch.cat((x[:batch_size], x_a), 0)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.layer_n = len(features)
        self.classifier1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.chan_ex = CUTI()

    def forward(self, x, y=None, choice=0):
        if y == None:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)
            return x
        elif choice % 2 == 0:
            input = torch.cat((x, y), 0)
            input = self.features(input)
            x, y = input.chunk(2, dim=0)

            x = x.view(x.size(0), -1)
            x = self.classifier1(x)

            y = y.view(y.size(0), -1)
            y = self.classifier1(y)

            return x, y

        elif choice % 2 == 1:
            input = torch.cat((x, y), 0)
            input = self.chan_ex(self.features[:3](input))
            input = self.chan_ex(self.features[3:6](input))
            input = self.chan_ex(self.features[6:11](input))
            input = self.chan_ex(self.features[11:16](input))
            input = self.chan_ex(self.features[16:](input))
            x, y = input.chunk(2, dim=0)

            x = x.view(x.size(0), -1)
            x = self.classifier1(x)

            y = y.view(y.size(0), -1)
            y = self.classifier1(y)

            return x, y


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in range(len(cfg)):
        v = cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict = False)
    return model


def validate_class(val_loader, model, epoch, num_class=10):
    model.eval()
    correct = 0
    total = 0
    c_class = [0 for i in range(num_class)]
    t_class = [0 for i in range(num_class)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels[:, 0], axis=1)
        correct += (predicted == true_label).sum().item()
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1

    acc = 100.0 * correct / total
    print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=acc))
    model.train()
    return acc


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train():
    num_classes = 10

    # Load source datasets
    if SOURCE == 'cifar':
        dataset1 = get_cifar_data()
    elif SOURCE == 'stl':
        dataset1 = get_stl_data()

    # Load target datasets
    if TARGET == 'cifar':
        dataset2 = get_cifar_data()
    elif TARGET == 'stl':
        dataset2 = get_stl_data()

    print('datasets loaded!')
    PATH = 'D:/WLY/Documents/NUAA/CVPR/github/' + SOURCE + '_to_' + TARGET + '.pth'

    datafile = Cus_Dataset(mode='train', dataset_1=dataset1, begin_ind1=0, size1=5000,
                                         dataset_2=dataset2, begin_ind2=0, size2=5000)
    datafile_val1 = Cus_Dataset(mode='val', dataset_1=dataset1, begin_ind1=5000, size1=1000)
    datafile_val2 = Cus_Dataset(mode='val', dataset_1=dataset2, begin_ind1=5000, size1=1000)
    valloader1 = DataLoader(datafile_val1, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader2 = DataLoader(datafile_val2, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    model = vgg19(pretrained=True, num_classes=num_classes)
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()
    
    for epoch in range(nepoch):
        for i, (img1, label1, img2, label2) in enumerate(dataloader):
            img1, label1, img2, label2 = img1.to(device), label1.to(device), img2.to(device), label2.to(device)
            
            img1.float()
            label1 = label1.float()
            img2.float()
            label2 = label2.float()
            
            out1, out2 = model(img1, img2, i)

            out1 = F.log_softmax(out1,dim=1)
            loss1 = criterion_KL(out1, label1)

            out2 = F.log_softmax(out2,dim=1)
            loss2 = criterion_KL(out2, label2)
            alpha = 0.1
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            loss = loss1 - loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        acc1 = validate_class(valloader1, model, epoch, num_class=num_classes)
        acc2 = validate_class(valloader2, model, epoch, num_class=num_classes)
        f = open(PATH.split('.')[0] + '.txt', "a+")
        f.write("epoch = {:02d}, acc_s = {:.3f}, acc_t = {:.3f}".format(epoch, acc1, acc2) + '\n')
        f.close()
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    setup_seed(SEED)
    train()

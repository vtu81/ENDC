'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np



class narrow_VGG(nn.Module):
    '''
    narrow_VGG model for constructing backdoor-chain 
    '''
    def __init__(self, features):

        super(narrow_VGG, self).__init__()

        self.features = features
        self.use_classifier = False
        # self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        # self.classifier = nn.Sequential(
        #     # nn.Linear(2, 1),
        #     # nn.ReLU(True),
        #     nn.Linear(1, 1),
        #     # nn.ReLU(True),
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(2, 1),
            # nn.ReLU(True),
            nn.Linear(1, 1),
            # nn.ReLU(True),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.use_classifier: x = self.classifier(x)
        # print(x.shape)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'narrow': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M1', 2, 2, 2, 'M', 2, 2, 2, 'M'],
    'mnist': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M1', 1, 1, 1, 'M', 1, 1, 1, 'M'],
    'fashionmnist': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M1', 1, 1, 1, 'M', 1, 1, 1, 'M'],
    'fashionmnist_small': [1, 1, 'M', 1, 1, 'M'],
    'fashionmnist_large': [2, 2, 'M', 2, 2, 'M', 2, 2, 2, 'M1', 2, 2, 2, 'M1', 2, 2, 2, 'M1'],
    # 'cifar10': [3, 3, 'M', 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M'],
    'cifar10': [2, 2, 'M', 2, 2, 'M', 2, 2, 2, 'M', 2, 2, 2, 'M', 2, 2, 1, 'M'],
}



def narrow_vgg16():
    return narrow_VGG(make_layers(cfg['narrow'], batch_norm=True))

def narrow_mnist_vgg():
    return narrow_VGG(make_layers(cfg['mnist'], batch_norm=False))

def narrow_fashionmnist_vgg():
    return narrow_VGG(make_layers(cfg['fashionmnist'], batch_norm=False))

def narrow_fashionmnist_small_vgg():
    model = narrow_VGG(make_layers(cfg['fashionmnist_small'], batch_norm=False))
    model.classifier = nn.Sequential(
        nn.Linear(49, 1),
        # nn.ReLU(True),
        # nn.Linear(1, 1),
        # nn.ReLU(True),
    )
    model.use_classifier = True
    return model

def narrow_fashionmnist_large_vgg():
    model = narrow_VGG(make_layers(cfg['fashionmnist_large'], batch_norm=False))
    model.classifier = nn.Sequential(
        nn.Linear(16, 1),
        # nn.ReLU(True),
        # nn.Linear(1, 1),
        # nn.ReLU(True),
    )
    model.use_classifier = True
    return model

def narrow_cifar10_vgg():
    model = narrow_VGG(make_layers(cfg['cifar10'], batch_norm=False))
    # model.classifier = nn.Sequential(
    #     nn.Linear(3, 1),
    #     nn.ReLU(True),
    #     nn.Linear(1, 1),
    #     nn.ReLU(True),
    # )
    # model.use_classifier = True
    return model

if __name__ == '__main__':
    # model = narrow_vgg16()
    # model = narrow_mnist_vgg()
    model = narrow_fashionmnist_vgg()
    x = torch.rand((3, 1, 28, 28))
    model(x)
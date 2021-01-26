'''
  code's author is leilei
'''

import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F

'''
    U_Net: original not based on vgg11 or vgg16
    only resnet has bias=False,so need you in write resnet notice bias=False
    batch_norm :is_training on pytorch is model.eval(); on tf is placeholder
'''

def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))

def upsample(in_features, out_features):
    shape = out_features.shape[2:]  # h w
    return F.upsample(in_features, size=shape, mode='bilinear', align_corners=True)

def concat(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)

class U_Net(nn.Module):
    def __init__(self, class_number=5, in_channels=3):
        super().__init__()
        # encoder
        self.conv1_1 = conv3x3_bn_relu(in_channels, 64)
        self.conv1_2 = conv3x3_bn_relu(64, 64)

        self.maxpool = nn.MaxPool2d(2, 2)  # only one for all

        self.conv2_1 = conv3x3_bn_relu(64, 128)
        self.conv2_2 = conv3x3_bn_relu(128, 128)

        self.conv3_1 = conv3x3_bn_relu(128, 256)
        self.conv3_2 = conv3x3_bn_relu(256, 256)

        self.conv4_1 = conv3x3_bn_relu(256, 512)
        self.conv4_2 = conv3x3_bn_relu(512, 512)

        self.conv5_1 = conv3x3_bn_relu(512, 1024)
        self.conv5_2 = conv3x3_bn_relu(1024, 1024)

        # decoder
        self.conv6 = conv3x3_bn_relu(1024, 512)
        self.conv6_1 = conv3x3_bn_relu(1024, 512)  ##
        self.conv6_2 = conv3x3_bn_relu(512, 512)

        self.conv7 = conv3x3_bn_relu(512, 256)
        self.conv7_1 = conv3x3_bn_relu(512, 256)  ##
        self.conv7_2 = conv3x3_bn_relu(256, 256)

        self.conv8 = conv3x3_bn_relu(256, 128)
        self.conv8_1 = conv3x3_bn_relu(256, 128)  ##
        self.conv8_2 = conv3x3_bn_relu(128, 128)

        self.conv9 = conv3x3_bn_relu(128, 64)
        self.conv9_1 = conv3x3_bn_relu(128, 64)  ##
        self.conv9_2 = conv3x3_bn_relu(64, 64)

        self.score = nn.Conv2d(64, class_number, 1, 1)

    def forward(self, x):
        # encoder
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.maxpool(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.maxpool(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        # decoder
        up6 = upsample(conv5_2, conv4_2)
        conv6 = self.conv6(up6)
        merge6 = concat(conv6, conv4_2)
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        up7 = upsample(conv6_2, conv3_2)
        conv7 = self.conv7(up7)
        merge7 = concat(conv7, conv3_2)
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        up8 = upsample(conv7_2, conv2_2)
        conv8 = self.conv8(up8)
        merge8 = concat(conv8, conv2_2)
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        up9 = upsample(conv8_2, conv1_2)
        conv9 = self.conv9(up9)
        merge9 = concat(conv9, conv1_2)
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)

        score = self.score(conv9_2)

        return score


def unet_orig(class_number, in_channels=3):
    model = U_Net(class_number, in_channels)
    return model
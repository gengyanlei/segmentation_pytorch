"""
@author: leilei
"""

import torch
from torch import nn
import torchvision
import torch.nn.functional as F

'''
Note:
    PSPNet: first conv7k_2s modify conv3k_2s/conv3k_1s/conv3k_1s(3 layers)
    each downsample block: first conv1k_1s modify conv1k_2s; second conv3k_2s modify conv3k_1s
    layer1: no downsample
    layer2: downsample
    layer3: no downsample; each block the second conv3x3 modify atros_conv3k_2r
    layer4: no downsample; each block the second conv3x3 modify atros_conv3k_4r
    Note: Resnet no bias,so bias = False

'''

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding bn relu and no bias"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                   bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding and no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution and no bias; downsample 1/stride"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def atrous_conv3x3(in_planes, out_planes, rate=1, padding=1, stride=1):
    """3x3  atrous convolution and no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                     dilation=rate, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None):
        '''
        pspnet conv1_3's num_output=128 not 64 so we modify some code
        first_inplanes: only layer1 not same (conv1_3)128 != (layer1-block1-conv1k_1s)64
        '''
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)  ####
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = atrous_conv3x3(planes, planes, rate, padding)  ####
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # only first layer1 block in_channel different
        if (first_inplanes != inplanes) and (downsample is not None):
            self.conv1 = conv1x1(first_inplanes, planes, stride)
            self.downsample = nn.Sequential(conv1x1(first_inplanes, planes * self.expansion, stride),
                                            nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SppBlock(nn.Module):
    # no bias
    def __init__(self, level, in_channel=2048, out_numput=512):
        super().__init__()
        self.level = level
        self.convblock = nn.Sequential(conv1x1(in_channel, out_numput),
                                       nn.BatchNorm2d(out_numput), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, output_size=(self.level, self.level))  # average pool
        x = self.convblock(x)
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        return x


class SppBlock1(nn.Module):
    # no bias k=10/20/30/60
    def __init__(self, level, k, s, in_channel=2048, out_numput=512):
        super().__init__()
        self.level = level
        self.avgpool = nn.AvgPool2d(k, s)
        self.convblock = nn.Sequential(conv1x1(in_channel, out_numput),
                                       nn.BatchNorm2d(out_numput), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x = self.avgpool(x)
        x = self.convblock(x)
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        return x


class SPP(nn.Module):
    def __init__(self, in_channel=2048):
        super().__init__()
        self.spp1 = SppBlock(level=1, in_channel=in_channel)
        self.spp2 = SppBlock(level=2, in_channel=in_channel)
        self.spp3 = SppBlock(level=3, in_channel=in_channel)
        self.spp6 = SppBlock(level=6, in_channel=in_channel)

    def forward(self, x):
        # x 2048 num_output
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x6 = self.spp6(x)
        out = torch.cat([x, x1, x2, x3, x6], dim=1)

        return out


class PSPNet(nn.Module):
    def __init__(self, block, layers, class_number, dropout_rate=0.2, in_channel=3):
        super().__init__()
        self.inplanes = 64
        self.conv1_1 = conv3x3_bn_relu(in_channel, 64, stride=2)
        self.conv1_2 = conv3x3_bn_relu(64, 64)
        self.conv1_3 = conv3x3_bn_relu(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, 64, layers[0])  # 64 / 256
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)  # 128 / 512
        self.layer3 = self._make_layer(block, 512, 256, layers[2], rate=2, padding=2)  # 256 / 1024
        self.layer4 = self._make_layer(block, 1024, 512, layers[2], rate=4, padding=4)  # 512 / 2048

        self.spp = SPP(in_channel=2048)

        self.conv5_4 = conv3x3_bn_relu(2048 + 512 * 4, 512)  ##if you want modify in_channel, need your own modify##

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv6 = nn.Conv2d(512, class_number, 1, 1)

        ''' init weight '''
        print('## init weight ##')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # no convtranspose linear

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.conv5_4(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = F.upsample(x, size, mode='bilinear', align_corners=True)

        return x

    '''first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None'''

    def _make_layer(self, block, first_inplanes, planes, blocks, rate=1, padding=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # with down stride same
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(first_inplanes, self.inplanes, planes, rate, padding, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, planes, rate, padding))

        return nn.Sequential(*layers)


def pspnet(class_number, dropout_rate=1):
    model = PSPNet(Bottleneck, layers=[3, 4, 6, 3], class_number=class_number, dropout_rate=dropout_rate)
    return model

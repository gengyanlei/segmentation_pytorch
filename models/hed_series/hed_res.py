"""
author: LeiLei
"""

'''
HED是基于VGG16构建的，
基于VGG16 或者resnet34 系列进行类似HED网络结构构建。
核心：就是3或者4或者5个尺度变化，而且pytorch也有每个残差block的类 输出属性，直接调用。
类似tensorflow的slim的output_collections。
'''
import torch
import torchvision
from torch import nn

class HED_res34(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, class_number=2):
        super().__init__()
        encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.pool = nn.MaxPool2d(3, 2, 1)

        # start
        self.start = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)  # 128*128
        self.d_convs = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.scores = nn.UpsamplingBilinear2d(scale_factor=2)  # 256*256

        self.layer1 = encoder.layer1  # 64*64
        self.d_conv1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.score1 = nn.UpsamplingBilinear2d(scale_factor=4)  # 256*256

        self.layer2 = encoder.layer2  # 32*32
        self.d_conv2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.score2 = nn.UpsamplingBilinear2d(scale_factor=8)  # 256*256

        self.layer3 = encoder.layer3  # 16*16
        self.d_conv3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.score3 = nn.UpsamplingBilinear2d(scale_factor=16)  # 256*256

        self.layer4 = encoder.layer4  # 8*8
        self.d_conv4 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score4 = nn.UpsamplingBilinear2d(scale_factor=32)  # 256*256

        self.score = nn.Conv2d(5, class_number, 1, 1)  # No relu loss_func has softmax

    def forward(self, x):
        x = self.start(x)
        s_x = self.d_convs(x)
        ss = self.scores(s_x)
        x = self.pool(x)

        x = self.layer1(x)
        s_x = self.d_conv1(x)
        s1 = self.score1(s_x)

        x = self.layer2(x)
        s_x = self.d_conv2(x)
        s2 = self.score2(s_x)

        x = self.layer3(x)
        s_x = self.d_conv3(x)
        s3 = self.score3(s_x)

        x = self.layer4(x)
        s_x = self.d_conv4(x)
        s4 = self.score4(s_x)

        score = self.score(torch.cat([s1, s2, s3, s4, ss], axis=1))

        return score


# hed2 = HED_res34()
# print(hed2)
# print(hed2.state_dict().keys())
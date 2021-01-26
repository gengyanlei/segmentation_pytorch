"""
author: LeiLei
"""

'''
HED是基于VGG16构建的，
基于VGG16 或者resnet34 系列进行类似HED网络结构构建。
核心：就是3或者4或者5个尺度变化，而且pytorch也有每个残差block的类 输出属性，直接调用。
类似tensorflow的slim的output_collections。

VGG16 network
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''
import torch
import torchvision
from torch import nn

# input size [256,256] or [512,512]
# 基于vgg16 hed
class HED_vgg16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, class_number=2):
        # Here is the function part, with no braces ()
        super().__init__()
        encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = encoder[0:4]
        self.score1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))  # 256*256

        self.conv2 = encoder[5:9]
        self.d_conv2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))  # 128*128
        self.score2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 256*256

        self.conv3 = encoder[10:16]
        self.d_conv3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))  # 64*64
        self.score3 = nn.UpsamplingBilinear2d(scale_factor=4)  # 256*256

        self.conv4 = encoder[17:23]
        self.d_conv4 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))  # 32*32
        self.score4 = nn.UpsamplingBilinear2d(scale_factor=8)  # 256*256

        self.conv5 = encoder[24:30]
        self.d_conv5 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))  # 16*16
        self.score5 = nn.UpsamplingBilinear2d(scale_factor=16)  # 256*256

        self.score = nn.Conv2d(5, class_number, 1, 1)  # No relu

    def forward(self, x):
        # Here is the part that calculates the return value
        x = self.conv1(x)
        s1 = self.score1(x)
        x = self.pool(x)

        x = self.conv2(x)
        s_x = self.d_conv2(x)
        s2 = self.score2(s_x)
        x = self.pool(x)

        x = self.conv3(x)
        s_x = self.d_conv3(x)
        s3 = self.score3(s_x)
        x = self.pool(x)

        x = self.conv3(x)
        s_x = self.d_conv4(x)
        s4 = self.score4(s_x)
        x = self.pool(x)

        x = self.conv5(x)
        s_x = self.d_conv5(x)
        s5 = self.score5(s_x)

        score = self.score(torch.cat([s1, s2, s3, s4, s5], axis=1))

        return score


''' you need to write softmax after model and predict output by yourself '''
# hed1 = HED_vgg16()
# print(hed1)
# print(hed1.state_dict().keys())
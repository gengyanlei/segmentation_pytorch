import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class HF_FCN(nn.Module):
    def __init__(self, class_number=2, pretrained=True, num_filters=32):
        super().__init__()
        encoder = torchvision.models.vgg16(pretrained=pretrained).features
        self.maxpool = encoder[4]

        self.conv1_1 = encoder[0:2]
        self.dconv1_1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv1_2 = encoder[2:4]
        self.dconv1_2 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        # 1/2
        self.conv2_1 = encoder[5:7]
        self.dconv2_1 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv2_2 = encoder[7:9]
        self.dconv2_2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        # 1/4
        self.conv3_1 = encoder[10:12]
        self.dconv3_1 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv3_2 = encoder[12:14]
        self.dconv3_2 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv3_3 = encoder[14:16]
        self.dconv3_3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        # 1/8
        self.conv4_1 = encoder[17:19]
        self.dconv4_1 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv4_2 = encoder[19:21]
        self.dconv4_2 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv4_3 = encoder[21:23]
        self.dconv4_3 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        # 1/16
        self.conv5_1 = encoder[24:26]
        self.dconv5_1 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv5_2 = encoder[26:28]
        self.dconv5_2 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv5_3 = encoder[28:30]
        self.dconv5_3 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))

        self.score = nn.Conv2d(13, class_number, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        x = self.conv1_1(x)
        s1_1 = self.dconv1_1(x)
        x = self.conv1_2(x)
        s1_2 = self.dconv1_2(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        s = self.dconv2_1(x)  # first reduce out_channels then upsample
        s2_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv2_2(x)
        s = self.dconv2_2(x)
        s2_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        s = self.dconv3_1(x)
        s3_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv3_2(x)
        s = self.dconv3_2(x)
        s3_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv3_3(x)
        s = self.dconv3_3(x)
        s3_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        s = self.dconv4_1(x)
        s4_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv4_2(x)
        s = self.dconv4_2(x)
        s4_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv4_3(x)
        s = self.dconv4_3(x)
        s4_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)

        x = self.conv5_1(x)
        s = self.dconv5_1(x)
        s5_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv5_2(x)
        s = self.dconv5_2(x)
        s5_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv5_3(x)
        s = self.dconv5_3(x)
        s5_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)

        score = self.score(torch.cat([s1_1, s1_2,
                                      s2_1, s2_2,
                                      s3_1, s3_2, s3_3,
                                      s4_1, s4_2, s4_3,
                                      s5_1, s5_2, s5_3], dim=1))  # no relu
        return score


def hf_fcn(class_number=5, fine_tune=True):
    model = HF_FCN(class_number=class_number, pretrained=fine_tune)
    return model
"""
    code's author: leilei
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

'''
also fine-tune
deeplab_v3+ : pytorch resnet 18/34 Basicblock
                      resnet 50/101/152 Bottleneck
            this is not original deeplab_v3+, just be based on pytorch's resnet, so many different.
'''
class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self,in_channel=512, depth=256):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel,depth,1,1), nn.ReLU(inplace=True))
        
        self.atrous_block1  = nn.Sequential(nn.Conv2d(in_channel,depth,1,1),
                                            nn.ReLU(inplace=True))
        self.atrous_block6  = nn.Sequential(nn.Conv2d(in_channel,depth,3,1,padding=6,dilation=6),
                                            nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel,depth,3,1,padding=12,dilation=12),
                                            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel,depth,3,1,padding=18,dilation=18),
                                            nn.ReLU(inplace=True))
        
        self.conv_1x1_output= nn.Sequential(nn.Conv2d(depth*5,depth,1,1), nn.ReLU(inplace=True))
        
    def forward(self,x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        
        atrous_block1  = self.atrous_block1(x)
        
        atrous_block6  = self.atrous_block6(x)
        
        atrous_block12 = self.atrous_block12(x)
        
        atrous_block18 = self.atrous_block18(x)
        
        net = self.conv_1x1_output(torch.cat([image_features,atrous_block1,atrous_block6,
                                            atrous_block12,atrous_block18],dim=1))
        return net
        
class Deeplab_v3_plus(nn.Module):
    # in_channel = 3 fine-tune
    def __init__(self, class_number=5, fine_tune=True, backbone='resnet50'):
        super().__init__()
        # 可选择resnet系列不同大小的网络
        encoder = getattr(torchvision.models, backbone)(pretrained=fine_tune)
        self.start = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)

        self.maxpool = encoder.maxpool
        self.low_feature = nn.Sequential(nn.Conv2d(64,48,1,1),nn.ReLU(inplace=True)) # no bn, has bias and relu
        
        self.layer1 = encoder.layer1  # 256
        self.layer2 = encoder.layer2  # 512
        self.layer3 = encoder.layer3  # 1024
        self.layer4 = encoder.layer4  # 2048
        
        self.aspp = ASPP(in_channel=self.layer4[-1].conv1.in_channels, depth=256)
        
        self.conv_cat = nn.Sequential(nn.Conv2d(256+48,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.conv_cat1 = nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.conv_cat2 = nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.score = nn.Conv2d(256,class_number,1,1)# no relu and first conv then upsample, reduce memory
        
    def forward(self,x):
        size1 = x.shape[2:]  # need upsample input size
        x  = self.start(x)
        xm = self.maxpool(x)
        
        x = self.layer1(xm)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        
        low_feature = self.low_feature(xm)
        size2 = low_feature.shape[2:]
        decoder_feature = F.upsample(x,size=size2,mode='bilinear',align_corners=True)
        
        conv_cat  = self.conv_cat( torch.cat([low_feature,decoder_feature],dim=1) )
        conv_cat1 = self.conv_cat1(conv_cat)
        conv_cat2 = self.conv_cat2(conv_cat1)
        score_small = self.score(conv_cat2)
        score = F.upsample(score_small,size=size1,mode='bilinear',align_corners=True)
        
        return score
        

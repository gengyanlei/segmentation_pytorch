import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F

'''
deeplab_v3+ : pytorch resnet 18/34 Basicblock
                      resnet 50/101/152 Bottleneck
'''
class ASPP(nn.Module):
    def __init__(self,in_channel=512,depth=256):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean=nn.AdaptiveAvgPool2d((1,1))
        self.conv=nn.Conv2d(in_channel,depth,1,1)
        # k=1 s=1 no pad
        self.atrous_block1=nn.Conv2d(in_channel,depth,1,1)
        self.atrous_block6=nn.Conv2d(in_channel,depth,3,1,padding=6,dilation=6)
        self.atrous_block12=nn.Conv2d(in_channel,depth,3,1,padding=12,dilation=12)
        self.atrous_block18=nn.Conv2d(in_channel,depth,3,1,padding=18,dilation=18)
        
        self.conv_1x1_output=nn.Conv2d(depth*5,depth,1,1)
        
    def forward(self,x):
        size=x.shape[2:]

        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features=F.upsample(image_features,size=size,mode='bilinear',align_corners=True)
        
        atrous_block1=self.atrous_block1(x)
        
        atrous_block6=self.atrous_block6(x)
        
        atrous_block12=self.atrous_block12(x)
        
        atrous_block18=self.atrous_block18(x)
        
        net=self.conv_1x1_output(torch.cat([image_features,atrous_block1,atrous_block6,
                                            atrous_block12,atrous_block18],dim=1))
        return net
        
class Deeplab_v3(nn.Module):
    def __init__(self,class_number=5):
        super().__init__()
        encoder=torchvision.models.resnet50(pretrained=True)
        
        #self.start=nn.Sequential(encoder.conv1,encoder.bn1,
        #                         encoder.relu)
        # (7,2,3)=>(3,1,1)+(3,1,1)+(3,2,1) 修改之处  这里会使网络深，参数减少。
        self.start=nn.Sequential(nn.Conv2d(3,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),\
                                 nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),\
                                 nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.maxpool=encoder.maxpool 
        self.layer1=encoder.layer1
        ##########
        self.encoder_feature=nn.Sequential(nn.Conv2d(256,128,1,1),nn.ReLU(inplace=True))
        
        self.layer2=encoder.layer2#512
        self.layer3=encoder.layer3#1024
        self.layer4=encoder.layer4#2048
        self.aspp=ASPP(in_channel=2048)
        self.conv_a=nn.Sequential(nn.Conv2d(256,256,1,1),nn.ReLU(inplace=True))
        self.conv_cat=nn.Sequential(nn.Conv2d(256+128,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.conv_cat1=nn.Sequential(nn.Conv2d(256,256,3,1,padding=1),nn.ReLU(inplace=True))
        self.score=nn.Conv2d(256,class_number,1,1)# no relu
        
    def forward(self,x):
        size1=x.shape[2:]# need upsample input size
        x=self.start(x)
        x=self.maxpool(x)
        xm=self.layer1(x)
        x=self.layer2(xm)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.aspp(x)
        x=self.conv_a(x)
        encoder_feature=self.encoder_feature(xm)
        size2=encoder_feature.shape[2:]
        decoder_feature=F.upsample(x,size=size2,mode='bilinear',align_corners=True)
        
        conv_cat=self.conv_cat(torch.cat([encoder_feature,decoder_feature],dim=1))
        conv_cat1=self.conv_cat1(conv_cat)
        score_conv=F.upsample(conv_cat1,size=size1,mode='bilinear',align_corners=True)#偶然发现，upsample可以放在self.score后面，减少训练时显存使用
        ##建议以后凡是最后一层前面是upsample，最后一层为转类别概率层，将upsample层和最后一层互换位置，减少使用显存##
        score=self.score(score_conv)
        return score
        
        
def deeplab_v3_50(class_number=5):
    model=Deeplab_v3(class_number=class_number)
    return model

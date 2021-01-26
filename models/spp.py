'''
sppnet base on vgg16
input size random ; but batch size need set to be 1 , and we don't use 'batch_size' .
'''

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class SppNet(nn.Module):
    def __init__(self, batch_size=1, out_pool_size=[1, 2, 4], class_number=2):
        super().__init__()
        # use already written network , eg vgg16
        vgg = torchvision.models.vgg16(pretrained=False).features[:-1]
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size
        # encoder
        self.encoder = vgg
        # spp if spp is a class , so create network ,it appear (spp)
        self.spp = self.make_spp(batch_size=batch_size, out_pool_size=out_pool_size)
        # FC
        sum0 = 0
        for i in out_pool_size:
            sum0 += i ** 2
        self.fc = nn.Sequential(nn.Linear(512 * sum0, 1024), nn.ReLU(inplace=True))
        self.score = nn.Linear(1024, class_number)

    def make_spp(self, batch_size=1, out_pool_size=[1, 2, 4]):
        func = []
        for i in range(len(out_pool_size)):
            func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i], out_pool_size[i])))
        return func

    def forward(self, x):
        assert x.shape[0] == 1, 'batch size need to set to be 1'
        encoder = self.encoder(x)
        spp = []
        for i in range(len(self.out_pool_size)):
            spp.append(self.spp[i](encoder).view(self.batch_size, -1))
        fc = self.fc(torch.cat(spp, dim=1))
        score = self.score(fc)
        return score


''' or another '''

class SppNet1(nn.Module):
    def __init__(self, batch_size=1, out_pool_size=[1, 2, 4], class_number=2):
        super().__init__()
        # use already written network , eg vgg16
        vgg = torchvision.models.vgg16(pretrained=False).features[:-1]
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size
        # encoder
        self.encoder = vgg
        # FC
        sum0 = 0
        for i in out_pool_size:
            sum0 += i ** 2
        self.fc = nn.Sequential(nn.Linear(512 * sum0, 1024), nn.ReLU(inplace=True))
        self.score = nn.Linear(1024, class_number)

    def forward(self, x):
        assert x.shape[0] == 1, 'batch size need to set to be 1'
        encoder = self.encoder(x)
        spp = []
        for i in range(len(self.out_pool_size)):
            spp.append(F.adaptive_avg_pool2d(encoder, output_size=(self.out_pool_size[i], self.out_pool_size[i])).view(
                self.batch_size, -1))
        fc = self.fc(torch.cat(spp, dim=1))
        score = self.score(fc)
        return score


# spp = SppNet(class_number=2)
# print(spp)
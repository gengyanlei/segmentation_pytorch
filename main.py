# the author is leilei
import os
import cv2 # bgr
import torch
import random
import torchvision
from torchvision import transforms
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image#######pytorch 默认读取图片########

from deeplab_v3_50 import deeplab_v3_50
import batch_data

# Hyper parameter 
batch_size=16
class_number=5
lr=2.5e-4
power=0.9
weight_decay=5e-4
max_iter=20000

dataset_path=r'/home/*/hdf5/f1.hdf5'

save_path=r'/home/*/model'
loss_s_path=os.path.join(save_path,'loss.npy')
model_s_path=os.path.join(save_path,'model.pth')
loss_s_figure=os.path.join(save_path,'loss.tif')
# some functions

def lr_poly(base_lr,iters,max_iter,power):
    return base_lr*((1-float(iters)/max_iter)**power)
def adjust_lr(optimizer,base_lr,iters,max_iter,power):
    lr=lr_poly(base_lr,iters,max_iter,power)
    optimizer.param_groups[0]['lr']=lr
    if len(optimizer.param_groups)>1:
        optimizer.param_groups[1]['lr']=lr*10
def get_1x_params(net):      ########详细教程请看本人的Pytorch-Tutorial-mnist##########
    b=[net.conv1,net.conv2,net.conv3,net.conv4,net.conv5]
    for i in b:
        for j in i.modules():
            for k in j.parameters():
                yield k
def get_10x_params(net):      ########详细教程请看本人的Pytorch-Tutorial-mnist##########
    b=[net.center,net.dec5,net.dec4,net.dec3,net.dec2,net.dec1,net.score]
    for i in b:
        for j in i.modules():
            for k in j.parameters():
                yield k
# loss func
softmax_loss=nn.CrossEntropyLoss().cuda()# multi class

# dataset
img_transform=transforms.ToTensor()
dataset=batch_data.Data(dataset_path,transform=img_transform)
trainloader=data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=3)
trainloader_iter=enumerate(trainloader)

model=deeplab_v3_50(class_number)

# fine-tune
#new_params=model.state_dict()
#pretrain_dict=torch.load(r'/home/*/model/model.pth')
#pretrain_dict={k:v for k,v in pretrain_dict.items() if k in new_params and v.size()==new_params[k].size()}# default k in m m.keys
#new_params.update(pretrain_dict)
#model.load_state_dict(new_params)

model.train()  # 与.cuda() 顺序不是固定的
model.cuda()

optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=weight_decay)
optimizer.zero_grad()

# start train
loss_npy=[]
loss_npy_10=[]
for iters in range(max_iter):
    loss_seg_v=0
    
    optimizer.zero_grad()
    adjust_lr(optimizer,lr,iters,max_iter,power)

    # train source
    try:
        _,batch=next(trainloader_iter)
    except:
        trainloader_iter=enumerate(trainloader)
        _,batch=next(trainloader_iter)
    
    images,labels=batch
    images=Variable(images).cuda()
    labels=Variable(labels).cuda()
    
    pred=model(images)
    loss_seg=softmax_loss(pred,labels)
    loss_seg.backward()
    loss_seg_v+=loss_seg.data.cpu().numpy().item()
    loss_npy.append(loss_seg_v)
    if len(loss_npy)==10:
        loss_npy_10.append(np.mean(loss_npy))
        loss_npy=[]
    optimizer.step()
    
    # output loss value
    print('/riter=%d , loss_seg=%.2f '%(iters,loss_seg_v),end='',flush=True)
    # save model
    if iters%1000==0 and iters!=0:
        # test image
        '''
        model.eval()  # 建议每个epoch训练完后，再进行整个epoch的测试精度计算，同时注意trian eval模式。也可以使用 with torch.no_grad():
        test_path=r'...'
        pre_path=r'...'
        names=os.listdir(test_path)
        for i in range(len(names)):
            img=Image.open(os.path.join(test_path,names[i]))
            r,g,b=img.split()
            img=Image.merge('RGB',(b,g,r))
            img_=img_transform(img)
            img_=torch.unsqueeze(img_,dim=0)
            image=Variable(img_).cuda()
            predict=model(image)
            P=torch.max(predict,1)[1].cuda().data.cpu().numpy()[0]
            P=np.uint8(P)
            cv2.imwrite(os.path.join(pre_path,names[i]),P)
        '''
        np.save(loss_s_path,loss_npy_10)
        torch.save(model.state_dict(),model_s_path)
        model.train()
np.save(loss_s_path,loss_npy_10)
torch.save(model.state_dict(),model_s_path)

# show loss figure
plt.figure()
plt.title('Loss Change')
plt.xlabel('iters')
plt.ylabel('loss value')
plt.plot(loss_npy_10,'-b',linewidth=1)
plt.savefig(loss_s_figure)











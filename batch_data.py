import cv2
import h5py
import torch
import random
import torchvision
from torchvision import transforms
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils import data

class Data(data.Dataset):
    def __init__(self,dataset_path,transform=None,augmentation=True):
        self.hdf5=h5py.File(dataset_path,mode='r')
        self.image=self.hdf5['image']
        self.label=self.hdf5['label']# not one-hot
        self.transform=transform# h*w*c => c*h*w and normlize[0 1]
        self.augmentation=augmentation
    def __len__(self):
        return self.image.shape[0]
    def data_augmentation(self,image,label):
        randint=random.randint(1,8)
        if randint==1:# left-right flip
            image=cv2.flip(image,1)
            label=cv2.flip(label,1)
        elif randint==2:# up-down-flip
            image=cv2.flip(image,0)
            label=cv2.flip(label,0)
        elif randint==3:# rotation 90 first width and then hight
            M=cv2.getRotationMatrix2D((image.shape[1]//2,image.shape[0]//2),90,1.0)
            image=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
            label=cv2.warpAffine(label,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        elif randint==4:# rotation 270
            M=cv2.getRotationMatrix2D((image.shape[1]//2,image.shape[0]//2),270,1.0)
            image=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
            label=cv2.warpAffine(label,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        return image,label
    def __getitem__(self,index):
        # here index is no N=1 direct h*w*c or h*w,then transform to c*h*w;dataloader auto add N=1 and batch_size
        img=self.image[index]# cv2 bgr remenber all example are the same as 'index' h*w*c
        lab=self.label[index].argmax(axis=-1) # no one-hot 
        if self.augmentation:
            img,lab=self.data_augmentation(img,lab)
        if self.transform is not None:
            img=self.transform(img)# only totensor and normlize h*w*c=>c*h*w
        
        return img,np.int64(lab) # lab need int64=long
    
    #########################################Data Augmentation#########################################################
    ''' Semantic segmentation '''
   # The input image and the label image need to be synchronized, such as rotation, flipping, and the like.

    ''' classification '''

    '''
        ## 建议，在图像分类中，输入一般是H=W的，而真实图片不一定是正方形，因此先加入padding再进行transform，加入padding采用cv2，
        ## 并且加入的padding值是边缘复制，代码如下：
        H, W, C = img.shape
        max_length = max(H,W)
        top_size, bottom_size, left_size, right_size = (int((max_length-H)/2), int(np.ceil((max_length-H)/2)), int((max_length-W)/2), int(np.ceil((max_length-W)/2)))
        replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        
        # transform
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(size=(self.shape[0], self.shape[1])),
                # 可继续加入 旋转 随机裁剪等操作，设置概率
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply( [transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4,
                                       hue=0.2)], p=0.5),  #特别注意这里，颜色变化基本上使得网络学习不到原始图像，因此对其设置p=0.5的概率执行此操作#
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.40

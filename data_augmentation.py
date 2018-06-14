'''
先处理图像保存到一个文件夹下，（数据增强先做，然后读取数据就不需要边读边数据增强了）
ADE数据集：151类。
'''
import os
import cv2
import random
import numpy as np

train_img_path=r'/home/*/*/Dataset/ADE/images/training'
train_lab_path=r'/home/*/*/Dataset/ADE/annotations/training'
#val_img_path=r'/home/*/ADE/images/validation'
#val_lab_path=r'/home/*/ADE/annotations/validation'
save_path=r'/home/*/*/Dataset/ADE/HDF5'

names=sorted(os.listdir(train_img_path))

N1=20206*5 # 一个epoch 总共N1张
num=20206 # 一共num张图片，然后每张图片做5次处理，保存。

s_img_path=r'/home/*/*/Dataset/ADE/HDF5/image'#save image
s_lab_path=r'/home/*/*/Dataset/ADE/HDF5/label'

def first_data_augmen(image,label):# first
    randint=random.randint(0,2)
    if randint==1:
        f_scale=0.5+random.randint(0,10)/10
        image=cv2.resize(image,(0,0),fx=f_scale,fy=f_scale)
        label=cv2.resize(label,(0,0),fx=f_scale,fy=f_scale,interpolation=cv2.INTER_NEAREST)
    else :
        image=image
        label=label
    return image,label

def final_data_augmen(image,label): # final
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
    
def middle_data_augmen(image,label):#middle
    H,W=label.shape
    if H>=256 and W>=256:
        # random crop
        h=random.randint(0,H-256)
        w=random.randint(0,W-256)
        img=image[h:h+256,w:w+256,:]
        lab=label[h:h+256,w:w+256]
    else :
        # less than 256 ,follow the minimal to 256，and the other to be int(256/min*max)
        if H<W:# H is min =>256
            image=cv2.resize(image,(int(W*256/H),256))# default INTER_LINEAR
            label=cv2.resize(label,(int(W*256/H),256),interpolation=cv2.INTER_NEAREST)
        else :# W<=H ,W is min =>256
            image=cv2.resize(image,(256,int(H*256/W)))
            label=cv2.resize(label,(256,int(H*256/W)),interpolation=cv2.INTER_NEAREST)
        H,W=label.shape
        h=random.randint(0,H-256)
        w=random.randint(0,W-256)
        img=image[h:h+256,w:w+256,:]
        lab=label[h:h+256,w:w+256]
    return img,lab

# process data
num_i=0
for i in range(num):
    image=cv2.imread(os.path.join(train_img_path,names[i]),-1)
    name=names[i].split('.')[0]+'.png'
    print(name)
    label=cv2.imread(os.path.join(train_lab_path,name),-1)
    for j in range(5):
        image,label=first_data_augmen(image,label)# random scale size
        image,label=middle_data_augmen(image,label)# random crop 
        img,lab=final_data_augmen(image,label)# random flip rotation or normal
        if lab.max()>150:
            print('########################')
            break
        cv2.imwrite(os.path.join(s_img_path,str(num_i)+'.jpg'),img)
        cv2.imwrite(os.path.join(s_lab_path,str(num_i)+'.png'),lab)

        num_i+=1
print(num_i==N1) 

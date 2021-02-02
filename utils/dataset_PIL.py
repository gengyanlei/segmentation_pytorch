import cv2
import torch
import random
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils import data
import torchvision.transforms.functional as tf

# TODO random flip, rotate, crop, noise, hue-brightness-contrast-saturation, zoom(in out), copy-paste?, mosaic?
# TODO GDAL imread any type image; -> .dataset_GDAL.py
'''
    author is leilei
    语义分割数据增强时，需将图像和标签图同时操作，对于旋转，偏移等操作，会引入黑边(均为0值)，
    将引入的黑边 视为1类，标签值默认为0，真实标签从1开始。
    图像采用BILINEAR，标签图采用NEAREST
    目前采用 torchvision.transforms.functional 的API，此api与PIL的数据增强操作是一致的，只要转成PIL，均采用uint8
    https://pytorch.org/docs/1.6.0/torchvision/transforms.html#functional-transforms
'''
class Augmentations:
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
        self.fill = 0  # image label fill=0，0对应黑边

    '''
    以下操作，均为单操作，不可组合！，所有的操作输出均需要resize至input_hw
    且 image为3 channel，label为1 channel
    且 输入均为RGB-3通道
    '''
    def random_rotate(self, image, label, angle=None):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :param angle:  None, list-float, tuple-float
        :return:  PIL
        '''
        if angle is None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        elif isinstance(angle, list) or isinstance(angle, tuple):
            angle = random.choice(angle)

        image = tf.rotate(image, angle)
        label = tf.rotate(label, angle)

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)

        return image, label

    def random_flip(self, image, label):
        if random.random() > 0.5:
            image = tf.hflip(image)
            label = tf.hflip(label)
        if random.random() < 0.5:
            image = tf.vflip(image)
            label = tf.vflip(label)

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)

        return image, label

    # zoom in
    def random_resize_crop(self, image, label, scale=(0.3, 1.0), ratio=(1, 1)):
        # 等价于 随即裁剪+resize至指定大小，大部分为放大操作；
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)  # 是在原图上 某个区域范围内(ratio控制区域长宽)随机裁剪
        image = tf.resized_crop(image, i, j, h, w, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resized_crop(label, i, j, h, w, self.input_hw, interpolation=Image.NEAREST)

        return image, label

    # zoom out
    def random_resize_minify(self, image, label, scale=(0.3, 1.0)):
        # 等价于 resize+padding(随机位置)，大部分为缩小操作
        in_hw = image.size[::-1]

        factor = transforms.RandomRotation.get_params(scale)  # 等比例缩放，也可不等比例
        size = (int(in_hw[0]*factor), int(in_hw[1]*factor))
        image = tf.resize(image, size, interpolation=Image.BILINEAR)
        label = tf.resize(label, size,  interpolation=Image.NEAREST)
        # pad
        top_bottom = (self.input_hw[0] - size[0])
        left_right = (self.input_hw[1] - size[1])

        top = top_bottom >> 1 if top_bottom > 0 else 0
        bottom = top_bottom - top if top_bottom > 0 else 0
        left = left_right >> 1 if left_right > 0 else 0
        right = left_right - left if left_right > 0 else 0

        tf.pad(image, (left, top, right, bottom), fill=self.fill, padding_mode='constant')
        # 黑边 默认成 0 类
        tf.pad(label, (left, top, right, bottom), fill=self.fill, padding_mode='constant')

        return image, label

    ''' 
    core function, Similar to cv2.warpAffine()
    # 可以将其它的所有操作 都基于此 进行，类似于 cv2的仿射变换矩阵;但是cv2默认是左上角，
    不能保证保持中心不变，除非最后有中心偏移操作！那么之前也应该有中心的某些操作
    可参考torchvision.transforms.functional -> _get_inverse_affine_matrix
    '''
    def random_affine(self, image, label):
        # 随机仿射(随机偏移，随机旋转，随机放缩等整合)
        # TODO
        if random.random() > 0.5:
            # 透视变换 RandomPerspective
            width, height = image.size
            startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.5)
            # 0值填充，仍是原始图像大小，需要resize
            image = tf.perspective(image, startpoints, endpoints, interpolation=Image.BICUBIC, fill=self.fill)
            label = tf.perspective(label, startpoints, endpoints, interpolation=Image.NEAREST, fill=self.fill)
        elif random.random() < 0.5:
            # 随机旋转-平移-缩放-错切 4种仿射变换 pytorch实现的是保持中心不变 不错切
            ret = transforms.RandomAffine.get_params(degrees=180, translate=(0.3, 0.3), scale=(0.3, 2),
                                                     shear=None, img_size=image.size)
            # angle, translations, scale, shear = ret
            # 0值填充，仍是原始图像大小，需要resize
            image = tf.affine(image, *ret, resample=0, fillcolor=self.fill)  # PIL.Image.NEAREST
            label = tf.affine(label, *ret, resample=0, fillcolor=self.fill)

        # 将图像处理成要求的大小
        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)

        return image, label

    def random_color_jitter(self, image, label, brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2):
        # 随机颜色增强，这里的随机是值，而非发生概率：transforms.RandomApply
        transforms_func = transforms.ColorJitter(brightness=brightness,
                                                 contrast=contrast,
                                                 saturation=saturation,
                                                 hue=hue)
        image = transforms_func(image)
        # label = label

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)

        return image, label

    # gassian noise
    def random_noise(self, image, label, noise_sigma=10):
        in_hw = image.size[::-1]
        noise = np.random.randn(in_hw) * noise_sigma

        image = np.array(image) + noise  # broadcast
        image = Image.fromarray(image, "RGB")

        image = tf.resize(image, self.input_hw, interpolation=Image.BILINEAR)
        label = tf.resize(label, self.input_hw, interpolation=Image.NEAREST)

        return image, label

# 数据读取类
class Dataset(data.Dataset):
    def __init__(self, ):
        pass








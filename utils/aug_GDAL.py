import os
import cv2
import math
import random
import torch
import numpy as np
# from osgeo import gdal
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

'''
    author is leilei
    语义分割数据增强时，需将图像和标签图同时操作，对于旋转，偏移等操作，会引入黑边(均为0值)，
    将引入的黑边 视为1类，标签值默认为0，真实标签从1开始。
    图像采用BILINEAR，标签图采用NEAREST
    采用GDAL库，进行读取任意通道(尤其是>=4通道的影像)，并结合cv2进行处理
    由于GDAL数据增强操作很麻烦，虽然有重采样等操作，但是接口文档不太友好，而且cv2对于float32也支持仿射变换
'''

class Gdal_Read:
    # 采用GDAL读取任意通道的影像(图像)
    def __init__(self):
        pass
    def read_img(self, filename, only_data=True):
        dataset = gdal.Open(filename)  # 打开文件

        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵 [channels, height, width] RGB的顺序

        im_data = im_data.transpose((1, 2, 0))  # [H,W,C]  # RGB顺序
        del dataset

        if only_data:
            return im_data
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64
        # cv2 对于int32-64报错，但是对于float32可以

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset
        return

class Augmentations_GDAL:
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
        self.fill = 0  # image label fill=0，0对应黑边
    '''
    以下操作，均为单操作，不可组合！，所有的操作输出均需要resize至input_hw
    且 image为多通道，label为1通道
    采用GDAL读取，但是数据增强采用cv2执行，cv2支持int16,float32，不支持int32格式
    image:[HWC], label:[HW]
    '''
    # TODO
    def random_rotate(self, image, label, angle=None):
        '''
        :param image:  GDALasArray(ndarray) uint8 or int16 or float32
        :param label:  cv2.imread uint8
        :param angle:  None, list-float, tuple-float
        :return:  PIL
        '''
        if angle is None:
            angle = transforms.RandomRotation.get_params([-180, 180])
        elif isinstance(angle, list) or isinstance(angle, tuple):
            angle = random.choice(angle)

        h, w = label.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)  # 尺度不变，中心旋转
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.fill)
        label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.fill)

        # resize
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    def random_flip(self, image, label):
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        # resize
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    # zoom in
    def random_resize_crop(self, image, label, scale=(0.3, 1.0), ratio=(1, 1)):
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = image[i:i+h, j:j+w]
        label = label[i:i+h, j:j+w]

        # resize
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    # zoom out
    def random_resize_minify(self, image, label, scale=(0.3, 1.0)):
        in_hw = label.shape[:2]
        factor = transforms.RandomRotation.get_params(scale)  # 等比例缩放，也可不等比例
        size = (int(in_hw[1] * factor), int(in_hw[0] * factor))  # (w,h)

        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        # pad
        top_bottom = (self.input_hw[0] - size[0])
        left_right = (self.input_hw[1] - size[1])

        top = top_bottom >> 1 if top_bottom > 0 else 0
        bottom = top_bottom - top if top_bottom > 0 else 0
        left = left_right >> 1 if left_right > 0 else 0
        right = left_right - left if left_right > 0 else 0

        image = cv2.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=self.fill)
        label = cv2.copyMakeBorder(label, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=self.fill)

        # resize
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    # core func
    def random_affine(self, image, label, perspective=0.0, degrees=0.373, scale=0.898, shear=0.602, translate=0.245):
        # 随机仿射(随机偏移，随机旋转，随机放缩等整合)
        height, width = image.shape[:2]

        # Center refer yolov5's mosaic aug
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees) / math.pi * 180  # 增加将弧度 转成角度
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation float，先中心偏移，再进行各种操作，然后将中心转移至原始位置左右，都是随机
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (M != np.eye(3)).any():  # image changed
            image = cv2.warpAffine(image, M[:2], dsize=self.input_hw[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=self.fill)
            label = cv2.warpAffine(label, M[:2], dsize=self.input_hw[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=self.fill)
        else:
            # 若未变换，则直接resize，这种概率很小
            image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    def random_color_jitter(self, image, label, brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2):
        # 随机颜色增强
        # TODO 多通道(>=4)的颜色增强 如何操作？

        return image, label

    # gassian noise TODO gassian-blur
    def random_noise(self, image, label, noise_sigma=10):
        in_hw = label.shape[:2]
        noise = (np.random.randn(in_hw) * noise_sigma).astype(image.dtype)  # +-
        image += noise  # broadcast

        # resize
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    def random_blur(self, image, label, kernel_size=(5,5)):
        assert len(kernel_size) == 2, "kernel size must be tuple and len()=2"
        image = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=0)

        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return image, label

    # def random_mosaic(self, image4, label4):
    #     # TODO mosaic data-aug
    #     # image9 label9
    #     pass
    #     return


class Transforms_GDAL(object):
    def __init__(self, input_hw=(256, 256)):
        self.aug_gdal = Augmentations_GDAL(input_hw)
        self.aug_funcs = [a for a in self.aug_gdal.__dir__() if not a.startswith('_') and a not in self.aug_gdal.__dict__]
        print(self.aug_funcs)

    def __call__(self, image, label):
        '''
        :param image:  PIL RGB uint8
        :param label:  PIL, uint8
        :return:  PIL
        '''
        aug_name = random.choice(self.aug_funcs)
        print(aug_name)  # 类实例后，读取数据时会不停的调用这个，每次都应该随机选择吧！
        image, label = getattr(self.aug_gdal, aug_name)(image, label)
        return image, label

class TestRescale(object):
    # test
    def __init__(self, input_hw=(256, 256)):
        self.input_hw = input_hw
    def __call__(self, image, label):
        '''
        :param image: ndarray
        :param label: ndarray uint8
        :return:
        '''
        image = cv2.resize(image, self.input_hw[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label

class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W). but no norm to 0-1
    def __call__(self, image, label):
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label

class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None, value_scale=255):
        # mean's type list or tuple
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)

        # equal to norm [0,1] then similar to pytorch's norm
        self.mean = [item * value_scale for item in mean]
        try:
            self.std = [item * value_scale for item in std]
        except:
            self.std = std

    def __call__(self, image, label):
        # tensor
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label

if __name__ == '__main__':
    runer = Gdal_Read()
    # jpg tiff 均可读取
    # im_proj, im_geotrans, im_data = runer.read_img(filename=r'F:\DataSets\jishi_toukui\1bc523b1-7bb4-4a14-9b32-5476f04c853f.jpg')
    im_proj, im_geotrans, im_data = runer.read_img(filename=r'D:\A145984.jpg')

    # image label 需要同时处理
    train_transforms = transforms.Compose([Transforms_GDAL(input_hw=(150, 150)),
                                           ToTensor(),  # /255 totensor
                                           Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                           ])
    test_transforms = transforms.Compose([TestRescale(input_hw=(150, 150)),
                                          ToTensor(),  # /255
                                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                          ])

    in_tensor = train_transforms(im_data)
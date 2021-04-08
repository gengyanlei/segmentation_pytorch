import cv2
import torch
import os
import numpy as np
from PIL import Image
from torch.utils import data

'''
    默认PIL读取图像，由于linux GDAL不友好，暂时不实现！
    root/
        train_txt
            /root/images/aaa.jpg | replace(images, labels)
        images/
            aaa.jpg
            bbb.jpg
        labels/
            aaa.jpg
            bbb.jpg
'''

# 数据读取类
class LoadDataset(data.Dataset):
    def __init__(self, txt_path, transform=None, is_gdal=False):
        assert transform is None, "transform不能为None"
        self.transform = transform
        self.is_gdal = is_gdal
        with open(txt_path, 'r') as f:
            self.image_paths = f.readlines()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item].strip()
        label_path = image_path.replace('images', 'labels')  # image in images/ folder, label in labels/ folder

        if not self.is_gdal:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 保证是3通道, 即使是1通道
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  #label图像本来就是1通道
            image = Image.fromarray(image[:, :, ::-1], 'RGB')  # bgr->rgb->PIL
            label = Image.fromarray(label)

        image, label = self.transform(image, label)

        return image, label


# 获取 train-test各自对应的transform
def get_transform(is_gdal=False, input_hw=(256,256), value_scale=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    if is_gdal:
        if value_scale is None:
            value_scale = 255
        assert isinstance(value_scale, int), "value_scale需要为int类型"
        from utils.aug_GDAL import Compose, Transforms_GDAL, TestRescale, ToTensor, Normalize
        train_transforms = Compose([Transforms_GDAL(input_hw=input_hw),
                                    ToTensor(),  # just hwc->chw
                                    Normalize(mean, std, value_scale),  # note value_scale
                                    ])
        test_transforms = Compose([TestRescale(input_hw=input_hw),
                                   ToTensor(),
                                   Normalize(mean, std, value_scale),
                                   ])

    else:
        from utils.aug_PIL import Compose, Transforms_PIL, TestRescale, ToTensor, Normalize
        train_transforms = Compose([Transforms_PIL(input_hw=input_hw),
                                    ToTensor(),  # /255 totensor
                                    Normalize(mean, std),
                                    ])
        test_transforms = Compose([TestRescale(input_hw=input_hw),
                                   ToTensor(),  # /255
                                   Normalize(mean, std),
                                   ])

    return train_transforms, test_transforms


def load_data(params):
    '''
    :param params:  configs/parameter.yaml pasred params
    :return:
    '''
    # transform param
    is_gdal = params['is_gdal']
    input_hw = params['input_hw']
    value_scale = params['value_scale']
    mean = params['mean']
    std = params['std']
    # data loader
    train_txt_path = params['train_txt_path']
    test_txt_path = params['test_txt_path']
    batch_size = params['batch_size']
    num_workers = params['num_workers']

    # transform
    train_transforms, test_transforms = get_transform(is_gdal, input_hw, value_scale, mean, std)
    # train
    train_dataset = LoadDataset(train_txt_path, train_transforms, is_gdal)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    # test
    test_dataset = LoadDataset(test_txt_path, test_transforms, is_gdal)
    test_loader = data.DataLoader(test_dataset, batch_size*2, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader




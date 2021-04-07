import cv2
# from osgeo import gdal
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




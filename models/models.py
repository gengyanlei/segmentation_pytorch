'''
    get all kinds of models
    获取网络均从此处获取
'''
from .deeplab_v3_plus import Deeplab_v3_plus


def deeplab_v3_plus(class_number=5, fine_tune=True, backbone='resnet50'):
    model = Deeplab_v3_plus(class_number=class_number, fine_tune=fine_tune, backbone=backbone)
    return model
## Semantic Segmentation Pytorch
```
    author is leilei
    Restart this project from 2017-10-01 
```

### Environment
```
    python: 3.6+
    ubuntu16.04 or 18.04
    pytorch 1.6+ (cuda10.2 docker)
    tensorboard 2.0+
```

### Note
+ If a black border is introduced, it will be regarded as one type, and the default is 0 !
+ label value is [1, N], 0 is black border class !

### How to Use

+ Train

+ Test

+ Dataset Details
```
root：
    images:  
    labels: 
    train.txt：
        /home/dataset/seg/images/train/aaa.jpg
        /home/dataset/seg/images/train/bbb.jpg
    test.txt：
        /home/dataset/seg/images/test/ccc.jpg
    
how to match images and labels?
    '/home/dataset/seg/images/train/aaa.jpg'.replace('images', 'labels')
    or
    '/home/dataset/seg/labels/train/aaa.jpg'.replace('.jpg', '.png')

data enhancement:
    random flip, rotate, crop, noise, 
    hue-brightness-contrast-saturation, zoom(in out), copy-paste?, mosaic?
```

### Support Network
- [x] deeplab_v3_plus(models/deeplab_v3_plus.py)
- [x] [pspnet](models/pspnet.py)
- [x] [unet](models/unet.py)
- [x] [spp-net](models/spp.py)
- [x] [HF_FCN](models/hed_series/hf_fcn_vgg16.py)
+ TODO
- [ ] torchvision.models.deeplab_v3
- [ ] [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)
- [ ] [U^2Net](https://github.com/NathanUA/U-2-Net)
- [ ] ...

### Data Aug
+ [data-augumentations](./readmes/data_aug.md)


### others
* [building-segmentation-dataset](https://github.com/gengyanlei/build_segmentation_dataset) 
* [reflective-clothes-detect-dataset](https://github.com/gengyanlei/reflective-clothes-detect)

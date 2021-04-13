## Semantic Segmentation Pytorch
```
    author is leilei
    Restart this project from 2017-10-01 
    TODO
    The latest version of the code cannot be executed and is still being updated.
```

### Environment
```
    python: 3.6+
    ubuntu16.04 or 18.04
    pytorch 1.6+ (cuda10.2 docker)
    tensorboard 2.0+
    scikit-learn 0.24.1
```

### **Note**
+ If a black border is introduced, it will be regarded as one type, and the default is 0 !
+ label value is [1, N], 0 is black border class !
+ Not supporting distributed(NCCL), just support DataParallel.

### Getting Started
+ [How to Use](./readmes/train_cusom.md)

### Demo
+ TODO

### Support Network
- [x] [deeplab_v3_plus](models/deeplab_v3_plus.py)
- [x] [pspnet](models/pspnet.py)
- [x] [unet](models/unet.py)
- [x] [spp-net](models/spp.py)
- [x] [HF_FCN](models/hed_series/hf_fcn_vgg16.py)
- [ ] [deeplab_v3](https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py)
- [ ] [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)
- [ ] [U^2Net](https://github.com/NathanUA/U-2-Net)
- [ ] ...

### Data Aug
+ [**data-augumentations**](./readmes/data_aug.md)
```
support 
    random zoom-in/out, random noise,
    random blur, random color-jitter(brightness-contrast-saturation-hue)
    random affine, random rotate, random flip
```

### Others
* [building-segmentation-dataset](https://github.com/gengyanlei/build_segmentation_dataset) 
* [reflective-clothes-detect-dataset](https://github.com/gengyanlei/reflective-clothes-detect)

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
```
### How to Use
+ Train

+ Test

+ Dataset
这里要更新一下图像格式存放等！！！！！！！！！！
```
root：
    images:  
    labels: 
    train.txt：
        /home/dataset/seg/images/aaa.jpg
        /home/dataset/seg/images/bbb.jpg
    test.txt：
        /home/dataset/seg/images/ccc.jpg
```

### Support Network
- [x] deeplab_v3_plus

+ TODO
- [ ] [pspnet](models/pspnet.py)
- [ ] [unet](models/unet.py)
- [ ] [spp-net](models/spp.py)
- [ ] [HF_FCN](models/hed_series/hf_fcn_vgg16.py)
- [ ] [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)
- [ ] [U^2Net](https://github.com/NathanUA/U-2-Net)
- [ ] ...

### others
* [building-segmentation-dataset](https://github.com/gengyanlei/build_segmentation_dataset) 
* [reflective-clothes-detect-dataset](https://github.com/gengyanlei/reflective-clothes-detect)

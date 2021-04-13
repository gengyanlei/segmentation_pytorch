### Getting Started

#### Train
+ You need to prepare the data in the format of 'Dataset Details'.
+ Modify the configuration file parameters [parameter.yaml](../configs/parameter.yaml).
+ Modify [main.py](../main.py) args's params and model-network-code.

<details>
  <summary>Figure Notes (click to expand)</summary>
  + ![modify1](./main_modify.jpg)
  + ![modify2](./param_modify.jpg)
</details>

#### Test
+ TODO 

#### Dataset Details
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
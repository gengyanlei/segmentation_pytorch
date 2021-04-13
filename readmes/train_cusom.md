### Train Custom Data


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
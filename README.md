# deeplab_v3
pytorch deeplab_v3+

声明：
  Pytorch deeplab_v3+
  
  pytorch : 0.4.0 ； python : 3.5

  采用pytorch复现deeplab_v3+，由于pytorch的resnet与tensorflow的resnet不太一样，
  主要是stride顺序不同。因此，按照tensorflow的顺序更改pytorch。


Pytorch 有自己的可视化工具：visdom  （pip install visdom即可）
主函数写的一般，没有采用argparse模块。

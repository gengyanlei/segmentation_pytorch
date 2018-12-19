# deeplab_v3
pytorch deeplab_v3+

注意：
    复现deeplab_v3_plus网络与论文有不同：论文有7个block，此处只是fine-tune pytorch自带的resnet；中间上采样处num_output为48，接在max_pool之后，此处和pytorch resnet有关，因此修改一下。仅供参考，如何重点使用pytorch已经训练的且写好的网络。

声明：
  Pytorch deeplab_v3_plus
  
  pytorch : 0.4.0 ； python : 3.5

Pytorch 有自己的可视化工具：visdom  （pip install visdom即可）
主函数写的一般，没有采用argparse模块。

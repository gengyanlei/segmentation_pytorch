'''
    author is leilei
'''
import os
import torch
import yaml
import math
import argparse
from torch import nn
from torch import optim
from models.models import *
from utils.dataset import load_data
from utils.util import init_seeds, check_path, increment_path
from utils.trainval import train
from torch.utils.tensorboard import SummaryWriter


def main(args):
    # read super parameters
    with open(args.cfg_path, 'r', encoding='utf-8') as f:
        param_dict = yaml.load(f, Loader=yaml.FullLoader)

    # creat save folder path
    save_dir = increment_path(args.project)  # str
    check_path(save_dir)
    param_dict['save_dir'] = save_dir  # update to param_dict
    param_dict['model_name'] = args.model_name  # update to param_dict
    # tensorboard
    tb_writer = SummaryWriter(save_dir)

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = param_dict['device']

    # data loader
    data_loader = load_data(params=param_dict)

    init_seeds(seed=1)  # activation cudnn
    model = deeplab_v3_plus(class_number=param_dict['class_number'], fine_tune=True, backbone='resnet50').cuda()
    continue_epoch = 0
    if args.resume:
        model_dict = model.state_dict()
        pretrained_file = torch.load(args.resume)
        pretrained_dict = pretrained_file['model'].float().state_dict()
        continue_epoch = pretrained_file['epoch']
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and v.size()==model_dict[k[7:]].size()}
        assert len(pretrained_dict) == len(model_dict), "Unsuccessful import weight"
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model = nn.DataParallel(model)  # keys add '.module', and has .module attribute

    # TODO 许多要增加的，先完成，再改善！先成v1版本，再弄v2版本
    if args.adam:
        optimizer = optim.Adam(model.module.parameters(), lr=param_dict['lr0'], betas=(param_dict['momentum'], 0.999), weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.module.parameters(), lr=param_dict['lr0'], momentum=param_dict['momentum'], weight_decay=5e-4)

    # set lr_scheduler  Cosine Annealing
    lf = lambda x: ((1 + math.cos(x * math.pi / param_dict['epoches'])) / 2) * (1 - param_dict['lrf']) + param_dict['lrf']  # cosine
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # no save scheduler params

    # train stage
    train(data_loader, model, optimizer, scheduler, tb_writer, param_dict, continue_epoch)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='deeplab_v3_plus-resnet50', help='model name')
    parser.add_argument('--project', type=str, default='./runs/exp', help='weight and summary... folder')
    parser.add_argument('--cfg_path', type=str, default='./configs/parameter.yaml', help='parameter config file')
    parser.add_argument('--resume', type=str, default='', help='resume most recent training')
    parser.add_argument('--adam', type=bool, default=False, help='Adam optimizer or SGD optimizer')

    args = parser.parse_args()
    main(args)

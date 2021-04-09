'''
    train-test stage
'''
import torch
from torch import nn

def train(data_loader, model, optimizer, scheduler, tb_writer, param_dict, continue_epoch):
    cross_entropy = nn.CrossEntropyLoss()

    # first update lr
    for epoch in range(0, continue_epoch):
        scheduler.step()

    best_acc = 0
    for epoch in range(continue_epoch, param_dict['epoches']):
        model.train()
        scheduler.step()
        train_acc = 0  # TODO mean image? or mean all image pixel?
        for step, data in enumerate(data_loader['train']):
            loss = 0
            images, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = cross_entropy(outputs, labels)

            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()

        # TODO train-test tensorboard summary

        # val stage
        if ((epoch - continue_epoch) % param_dict['test_interval'] == 0) and \
                (epoch - continue_epoch) != 0:

            test(data_loader['test'], model, param_dict)

def test(test_loader, model, param_dict):

    cross_entropy = nn.CrossEntropyLoss()

    model.train(False)  # = model.eval() restrict bn
    with torch.no_grad():
        acc = 0
        for step, data in enumerate(test_loader):
            images, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)





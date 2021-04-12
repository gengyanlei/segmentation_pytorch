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
        train_acc = 0
        train_loss = 0  # TODO add list to reduce code
        for step, data in enumerate(data_loader['train']):
            loss = 0
            images, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss += cross_entropy(outputs, labels)
            train_loss += loss.cpu().item()

            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()

        # TODO train-test tensorboard summary
        tb_writer.add_scalar('Loss/train_loss', train_loss/len(data_loader['train']), epoch)

        # val stage
        if ((epoch - continue_epoch) % param_dict['test_interval'] == 0) and \
                (epoch - continue_epoch) != 0:

            test_loss, test_acc = test(data_loader['test'], model, param_dict)
            tb_writer.add_scalar('Loss/test_loss', test_loss, epoch)
            # TODO add metrics

    return

def test(test_loader, model, param_dict):

    cross_entropy = nn.CrossEntropyLoss()  # 已经经过mean

    model.train(False)  # = model.eval() restrict bn
    with torch.no_grad():
        test_acc = 0
        test_loss = 0
        for step, data in enumerate(test_loader):
            images, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)

            test_loss += loss.cpu().item()

    return test_loss/len(test_loader), test_acc



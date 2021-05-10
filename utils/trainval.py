'''
    train-test stage
'''
import torch
import numpy as np
from torch import nn
from pathlib import Path
from utils.plots import plot_confusion_matrix
from utils.metrics import get_confusion_matrix, compute_acc_pr_iou
import torch.nn.functional as F

def train(data_loader, model, optimizer, scheduler, tb_writer, param_dict, continue_epoch):
    # weights folder create
    save_dir = Path(param_dict['save_dir']) / 'weights'
    save_dir.mkdir(parents=True, exist_ok=True)
    last = save_dir / 'last.pt'
    best = save_dir / 'best.pt'

    cross_entropy = nn.CrossEntropyLoss()

    # first update lr
    for epoch in range(0, continue_epoch):
        scheduler.step()

    best_fitness = 0
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
        if ((epoch - continue_epoch) % param_dict['test_interval'] == 0) and (epoch - continue_epoch) != 0:

            test_loss, test_indexs = test(data_loader['test'], model, param_dict)
            tb_writer.add_scalar('Loss/test_loss', test_loss, epoch)
            tags = ['Metrics/Accuracy', 'Metrics/Mean_Precision', 'Metrics/Mean_Recall', 'Metrics/Mean_IoU']
            for tag, index in zip(tags, test_indexs):
                tb_writer.add_scalar(tag, index, epoch)
            # TODO best_fitness=w1*acc+w2*precision+w3*recall+w4*mean_iou
            # save best weight
            if test_indexs[-1] > best_fitness:
                best_fitness = test_indexs[-1]
                torch.save({'model': model,
                            'epoch': epoch,
                            'model_name': param_dict['model_name'],
                            'optimizer': optimizer.state_dict(),
                            'best_fitness': best_fitness}, best)
        # save last weight
        torch.save({'model': model,
                    'epoch': epoch,
                    'model_name': param_dict['model_name'],
                    'optimizer': optimizer.state_dict(),
                    'best_fitness': best_fitness if param_dict['test_interval'] == 1 else None}, last)

    # end training, last delete epoch etl information
    torch.save({'model': model,
                'model_name': param_dict['model_name'],
                'optimizer': None}, last)

    return

def test(test_loader, model, param_dict):
    confusion_matrix = np.zeros([param_dict['class_number'], param_dict['class_number']], dtype=np.int64)

    cross_entropy = nn.CrossEntropyLoss()  # 已经经过mean

    model.train(False)  # = model.eval() restrict bn
    with torch.no_grad():
        test_acc = 0
        test_loss = 0
        for step, data in enumerate(test_loader):
            images, labels = data
            inputs = inputs.cuda()
            labels_cuda = labels.cuda()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels_cuda)
            # compute confusion matrix
            outputs_p = F.softmax(outputs, dim=1)  # [N,C,H,W] cuda
            P = torch.max(outputs_p, 1)[1].data.cpu().numpy()  # [N,H,W] numpy-cpu

            m = get_confusion_matrix(P, labels.data.numpy(), class_number=param_dict['class_number'])
            confusion_matrix += m

            test_loss += loss.cpu().item()
    # plot confusion_matrix and save
    plot_confusion_matrix(confusion_matrix, param_dict['save_dir'], param_dict['class_names'])

    acc, mean_precision, mean_recall, mean_iou = compute_acc_pr_iou(confusion_matrix)

    return test_loss/len(test_loader), (acc, mean_precision, mean_recall, mean_iou)



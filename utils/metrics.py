'''
    Semantic segmentation metrics functions
'''
import numpy as np
from sklearn import metrics

def get_confusion_matrix(predicts, labels, class_number):
    # predicts.shape == labels.shape
    confusion_matrix = metrics.confusion_matrix(labels.reshape([-1]), predicts.reshape([-1]), labels=range(class_number))
    return confusion_matrix

def compute_acc_pr_iou(confusion_matrix):
    # Calculate various indicators according to the confusion matrix.
    diag = np.diag(confusion_matrix)
    p_s = np.sum(confusion_matrix, axis=0)
    r_s = np.sum(confusion_matrix, axis=1)

    acc = np.sum(diag) / np.sum(confusion_matrix)
    mean_precision = np.mean(diag / (p_s + 1e-6))  # per class precison's mean value
    mean_recall = np.mean(diag / (r_s + 1e-6))  # per class recall's mean value
    mean_iou = np.mean(diag / (p_s + r_s - diag + 1e-6))

    return acc, round(mean_precision, 4), round(mean_recall, 4), round(mean_iou, 4)


if __name__ == '__main__':
    m = get_confusion_matrix(predicts=np.ones([10,3]), labels=np.ones([10,3]), class_number=5)
    acc, mean_precision, mean_recall, mean_iou = compute_acc_pr_iou(m)
    print(m)
    print(acc, mean_precision, mean_recall, mean_iou)



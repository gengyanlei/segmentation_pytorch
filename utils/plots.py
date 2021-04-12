import os
import numpy as np
import matplotlib.pyplot as plt

# support chinese
# refer https://blog.csdn.net/lucky__ing/article/details/78699198
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 混淆矩阵-可视化
def plot_confusion_matrix(confusion_matrix, save_dir, class_names):
    plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Reds)
    indices = range(len(confusion_matrix))  # confusion_matrix [N,N]
    plt.xticks(indices, list(class_names))
    plt.yticks(indices, list(class_names))

    plt.colorbar()

    plt.xlabel('Predict-Value')
    plt.ylabel('Ground-Truth')
    plt.title('Confusion-Matrix')

    for first_index in range(len(confusion_matrix)):
        for second_index in range(len(confusion_matrix[first_index])):
            plt.text(first_index, second_index, confusion_matrix[first_index][second_index])

    plt.plot()  # draw image
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    # plt.show()  # show off image

    return



if __name__ == '__main__':
    confusion_matrix = np.array([[90,10,5,1,1],[10,80,15,1,1],[20,5,80,1,1],[1,1,1,95,10],[1,1,1,1,100]])
    save_dir = r'/home/gengyanlei/'
    class_names = ['我', '你', '他', '它', '她']
    # class_names = ['wo', 'ni', 'ta', '1', '2']
    plot_confusion_matrix(confusion_matrix, save_dir, class_names)


'''
    demo for predict one image, and visualizate.
'''
import os
import cv2
import yaml
import torch
import random
import argparse
import numpy as np
from PIL import Image
from utils.util import check_path
import torchvision.transforms.functional as tf

img_formats = ['jpg', 'png', 'tif', 'jpeg']

def predict(args, save_img=True):
    # load parameter.yaml
    with open(args.cfg_path, 'r', encoding='utf-8') as f:
        param_dict = yaml.load(f, Loader=yaml.FullLoader)
    mean = param_dict['mean']
    std = param_dict['std']
    class_names = param_dict['class_names']

    # save dir check exist
    check_path(args.output)

    # load model
    device = torch.device("cuda:{}".format(args.device))
    model = torch.load(args.weights, map_location=device)['model'].eval()  # if fail, before eval() add .to(device)

    # label color
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    # imread image
    names = os.listdir(args.source)
    img_names = [name for name in names if name.split('.')[-1].lower() in img_formats]

    with torch.no_grad():
        for img_name in img_names:
            img = cv2.imread(os.path.join(args.source, img_name), cv2.IMREAD_COLOR)[..., ::-1]  # bgr->rgb
            img_ = Image.fromarray(img, mode="RGB")
            # ToTensor -> Normalize -> gpu  [1,3,H,W]
            img_norm = tf.normalize(tf.to_tensor(img_), mean, std).to(device)

            pred = model(img_norm)  # [1,Class_num,H,W]
            pred_ = pred.argmax(dim=1)[0].cpu().numpy()
            pred3 = np.zeros(pred_.shape+(3,), dtype=np.uint8)
            if save_img:
                for i in range(len(class_names)):
                    pred3[pred_==i] = colors[i]
                cv2.imwrite(os.path.join(args.output, img_name), pred3)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Semantic segmentation Predict")
    parser.add_argument("--weights", type=str, default=r'./runs/exp/weights/best.pt', help="weight's path")
    parser.add_argument("--source", type=str, default=r'./data/images', help="input source")
    parser.add_argument("--device", type=str, default='0', help="gpu id, suggest 1 gpu")
    parser.add_argument("--output", type=str, default=r'./outputs', help="save dir")
    # parser.add_argument("--img_size", type=int, default=512, help="input image size")  # Any size
    args = parser.parse_args()

    predict(args, save_img=True)

import os
from pathlib import Path
import random
import math
import numpy as np
import torch
from torch.backends import cudnn
'''
    一些辅助函数
'''


def RandomResizedCrop_get_params(img, scale, ratio):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image): Image to be cropped.  修改成 numpy格式
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    height, width = img.shape[:2]
    area = height * width

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if (in_ratio < min(ratio)):
        w = width
        h = int(round(w / min(ratio)))
    elif (in_ratio > max(ratio)):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def init_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)
    return

# 为什么不使用 所有GPU的呢？
def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
    return

def check_path(path):
    p = Path(path)
    if not p.exists():
        p.mkdir()
    return


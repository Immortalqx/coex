import re
import random
import numpy as np

import cv2
import torch
import torch.utils.data as data

from .stereo_albumentation import horizontal_flip
from . import preprocess


# airsim官方读取pfm的代码，返回结果是正确的；那个readpfm.py的结果不太对
def read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    pattern = r'^(\d+)\s(\d+)\s$'
    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(pattern, temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        temp_str += str(bytes.decode(file.readline(), encoding='utf-8'))
        dim_match = re.match(pattern, temp_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header: width, height cannot be found')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    file.close()

    # 限制深度的最大值，TODO 这里手动写一下吧，以后可以写到配置文件里
    data = np.clip(data, 0, 30)

    return data, scale


def default_loader(path):
    return cv2.imread(path)


def disparity_loader(path):
    return read_pfm(path)


class ImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, right_dispartity, training,
                 loader=default_loader, dploader=disparity_loader,
                 th=128, tw=128, load_raw=False):

        self.left = left
        self.right = right

        self.disp_L = left_disparity
        self.disp_R = right_dispartity

        self.loader = loader
        self.dploader = dploader

        self.th = th
        self.tw = tw

        self.training = training

        self.K = torch.Tensor(np.array([[320.0, 0.0, 320.0],
                                        [0.0, 320.0, 240.0],
                                        [0, 0, 1]]))

        self.load_raw = load_raw

    def __getitem__(self, index):
        batch = dict()

        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        K = self.K

        left_img = self.loader(left)
        right_img = self.loader(right)

        if self.load_raw:
            left_img_ = np.transpose(left_img, (2, 0, 1)).astype(np.float32)
            right_img_ = np.transpose(right_img, (2, 0, 1)).astype(np.float32)
            batch['imgLRaw'], batch['imgRRaw'] = left_img_, right_img_

        dataL, scaleL = self.dploader(disp_L)
        dataR, scaleR = self.dploader(disp_R)

        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        if self.training:
            left_img, right_img, dataL = horizontal_flip(left_img, right_img, dataL, dataR)

            h, w = left_img.shape[:2]

            x1 = random.randint(0, w - self.tw)
            y1 = random.randint(0, h - self.th)

            left_img = left_img[y1: y1 + self.th, x1: x1 + self.tw]
            right_img = right_img[y1: y1 + self.th, x1: x1 + self.tw]

            dataL = dataL[y1:y1 + self.th, x1:x1 + self.tw]

            img = {'left': left_img, 'right': right_img}

            left_img, right_img = img['left'], img['right']

            processed = preprocess.get_transform(augment=True)
            left_img = processed(left_img)
            right_img = processed(right_img)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['K'], batch['x1'], batch['y1'] = K, x1, y1

            return batch
        else:
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['K'] = K

            return batch

    def __len__(self):
        return len(self.left)

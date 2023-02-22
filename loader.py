# -*- CODING: UTF-8 -*-
# @time 2023/2/22 16:50
# @Author tyqqj
# @File loader.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

import os
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchio as tio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)

# def mha_loader(path):


# mha读取器
path = r'D:\Data\kitware_brains'
dic = [2, 3, 4, 6, 8]  # 要读取的编号

__all__ = ['mha_dataloader']


# 返回data和lable
def mha_loader(path):
    # 读取data
    data = []
    for i in range(0, 5):
        data_path = os.path.join(path, 'Normal-00' + str(dic[i]), 'MRA', 'Normal00' + str(dic[i]) + '-MRA.mha')
        data.append(tio.ScalarImage(data_path).data)
    data = np.array(data)
    # 读取label
    label = []
    for i in range(0, 5):
        label_path = os.path.join(path, 'Normal-00' + str(dic[i]), 'MRA', 'Normal00' + str(dic[i]) + '.mha')
        label.append(tio.ScalarImage(label_path).data)
    label = np.array(label)
    return data, label

dataset = mha_loader(path)
print(dataset[0][0].shape)


def mha_dataloader(path, batch_size, shuffle=True, num_workers=0):
    dataset = mha_loader(path)
    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.Crop((0, 0, 10, 30, 40, 40)),
    ]
    transform = tio.Compose(transforms)
    datasets = transform(dataset)
    return data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
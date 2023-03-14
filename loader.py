# -*- CODING: UTF-8 -*-
# @time 2023/2/22 16:50
# @Author tyqqj
# @File loader.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchio as tio
import glob
import SimpleITK as sitk
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)

# def mha_loader(path):


# mha读取器


################# 参数 #################
path = r'D:\Data\brains'
dic = [2, 3, 4, 6, 8]  # 要读取的编号

__all__ = ['mha_dataloader']

args = {
    'train_patch_size_x': 96,
    'train_patch_size_y': 96,
    'train_patch_size_z': 96,
}

patch_size = (args['train_patch_size_x'], args['train_patch_size_y'], args['train_patch_size_z'])


################# mha读取器 #################


# 输入路径，返回图像和标签路径的列表
# 要求：图像和标签的顺序和数量一致
def mha_dataloader(path, train=True):
    # 如果路径不存在train，报错
    if not os.path.exists(os.path.join(path, 'train')):
        raise FileNotFoundError("train folder not found in " + path)

    img = []
    lbl = []
    # 如果train为True，读取训练集，否则读取测试集
    if train:
        mha_path = os.path.join(path, 'train')
    else:
        mha_path = os.path.join(path, 'test')

    for file in glob.glob(os.path.join(mha_path, "image", "*.mha")):
        # print(file)
        # 读取一个mha文件
        file_name = os.path.basename(file)[:-4]  # 去掉后缀

        # 判断标签是否存在
        lable_file = glob.glob(os.path.join(mha_path, "label", file_name + "*"))
        if len(lable_file) == 0:
            print("image", file_name, "has no label")
            continue
        if len(lable_file) > 1:
            print("image", file_name, "has more than one label")
            continue
        # print(file_name)
        # print(os.path.basename(lable_file[0])[:-4])
        img.append(file)
        lbl.append(lable_file[0])

    return img, lbl


################# 随机裁剪 #################


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()
    dataset = ((dataset - mean) / std).astype(dtype)
    return dataset


def extractPatch(data, p_x, p_y, p_z, x, y, z):
    patch_rst = data[x - p_x // 2:x + p_x // 2,
                y - p_y // 2:y + p_y // 2,
                z - p_z // 2:z + p_z // 2]
    return patch_rst


def RandomPatchCrop(image, label, patch_in_size, patch_gd_size):
    if (patch_in_size[0] % patch_gd_size[0] != 0 or patch_in_size[1] % patch_gd_size[1] != 0 or patch_in_size[2] %
            patch_gd_size[2] != 0):
        sys.exit("patch_in_size must be divisible by patch_gd_size")
    if (patch_in_size[0] < patch_gd_size[0] or patch_in_size[1] < patch_gd_size[1] or patch_in_size[2] < patch_gd_size[
        2]):
        sys.exit("patch_in_size must be greater than patch_gd_size")

    # 生成随机数
    x = randint(patch_size[0] // 2, image.shape[0] - patch_gd_size[0] // 2)
    y = randint(patch_size[1] // 2, image.shape[1] - patch_gd_size[1] // 2)
    z = randint(patch_size[2] // 2, image.shape[2] - patch_gd_size[2] // 2)

    # 生成随机旋转
    r0 = randint(5, 10)
    r1 = randint(5, 10)
    r2 = randint(5, 10)
    patch_in = extractPatch(image, patch_in_size[0], patch_in_size[1], patch_in_size[2], x, y, z)
    patch_gd = extractPatch(label, patch_gd_size[0], patch_gd_size[1], patch_gd_size[2], x, y, z)

    # 旋转
    patch_in = np.rot90(patch_in, r0, (0, 1))
    patch_in = np.rot90(patch_in, r1, (1, 2))
    patch_in = np.rot90(patch_in, r2, (0, 2))

    patch_gd = np.rot90(patch_gd, r0, (0, 1))
    patch_gd = np.rot90(patch_gd, r1, (1, 2))
    patch_gd = np.rot90(patch_gd, r2, (0, 2))

    return patch_in, patch_gd


################# 数据集 #################
class mha_data(Dataset):
    def __init__(self, root_dir, train=True, rotate=40, flip=True, random_crop=True, scale1=512):
        self.root_dir = root_dir
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1

        self.img, self.lbl = mha_dataloader(self.root_dir, self.train)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img_path = self.img[index]
        lbl_path = self.lbl[index]

        # 读取图像和标签
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image).astype(np.float32)

        label = sitk.ReadImage(lbl_path)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # 检查图像和标签是否一致
        if image.shape != label.shape:
            # 红色字体提醒数据
            print("\033[1;31mimage", os.path.basename(img_path)[:-4], "and label are not consistent\033[0m")

        # print(image.shape, '\n', label.shape, '\n')

        img_patch, lbl_patch = RandomPatchCrop(image, label, patch_size, patch_size)

        # print(img_patch.shape, '\n', lbl_patch.shape, '\n')

        img_patch = standardization_intensity_normalization(img_patch, np.float32)

        # print(img_patch.shape, '\n', lbl_patch.shape, '\n')

        image = torch.from_numpy(np.ascontiguousarray(img_patch)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(lbl_patch)).unsqueeze(0)

        return image, label


# 测试
# print(mha_dataloader(path, train=True))

# 测试
# dataset = mha_data(path)
# print(dataset[0][0].shape, '\n', dataset[0][1].shape, '\n')
# print(dataset[0][0].shape, '\n', dataset[0][1].shape, '\n')

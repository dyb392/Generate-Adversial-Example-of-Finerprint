import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from skimage import io, transform


class fingerprintDataset(Dataset):
    def __init__(self, train=True, transform=None, test=None):
        '''
        :param train: 读取训练集/测试集
        :param transform: 标准化数据
        '''
        self.transform = transform
        self.train = train
        if train is True:
            filename = os.path.join("train.csv")
        else:
            if test==None:
                filename = os.path.join("test.csv")
            elif test==1:
                filename = os.path.join("test1.csv")
            elif test==2:
                filename = os.path.join("test2.csv")
            elif test==3:
                filename = os.path.join("test3.csv")
        self.image_label_list = pd.read_csv(filename, header=0, encoding="gbk")
        self.image_label_list = np.array(self.image_label_list)
        #print(self.image_label_list[0])
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()
 
        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()
 
    def __getitem__(self, index):
        #print("###########################")
        #print("index={}".format(index))
        image_name, label = self.image_label_list[index]
        #print(image_name)
        #print("###########################")
        if self.train is True:
            image_path = os.path.join(image_name)
        else:
            image_path = os.path.join(image_name)
        #print(image_path)
        #print("###########################")
        img = io.imread(image_path)
        #print(np.array(img).shape)
        label = np.array(label)
        #label = label.astype("long")
        if self.transform:
            img = self.transform(img)
        return img, label
 
    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len

# title                 :data_loader.py
# description           :It manages the tomographic datasets that will feed the Deep Learning models
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-08-05
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import pandas as pd
import numpy as np
import scipy.misc
import random
import os
from imageio import imread

from Network.data_utils import data_mean_value
import torch
from torch.utils.data import Dataset
import read_kneeDataset as kd


"""
Parameters to manage the tomographic datasets 

train_dim:      dimensions (h,w) of the cropped files used in the training stage. h and w must be smaller than the size of the original images.

"""

train_dim   = (128,128)


class Tomographic_Dataset(Dataset):

    #directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
    def __init__(self, csv_file, phase, n_class, crop=False, flip_rate=0., train_csv = ""):
        self.data      = pd.read_csv(csv_file)
        self.means     = data_mean_value(train_csv, 10)
        self.flip_rate = flip_rate
        self.n_class   = n_class

        if phase == 'train':
            self.crop      = crop
            self.flip_rate = 0.5
            self.new_h = train_dim[0]
            self.new_w = train_dim[1]
        else:
            self.crop = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code   = self.data.iloc[idx, 0]
        slice  = self.data.iloc[idx, 1]
        img        = kd.getMRISlice(code, int(slice))
        gt      = kd.getSegmentationSlice(kd.getSegmentationCode(code), int(slice))

        h, w = gt.shape
        label = np.zeros((self.n_class,h,w))
        for c in range(self.n_class):
            label[c][gt == c] = 1

        if self.crop:
            h, w, _  = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img -= self.means

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).float()


        _ , h, w = label.size()
        data_in = torch.zeros(1,h,w)
        data_in[0,:,:] = img

        sample = {'X': data_in, 'Y': label, 'scan': code, 'slice': slice}

        return sample





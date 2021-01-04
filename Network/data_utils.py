# title                 :data_utils.py
# description           :It creates a training-validation-test split and computes the mean gray value in the images of the training set
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-05-16
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# matplotlib_version    :3.0.3
# pilow_version         :6.0.0
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import os
import numpy as np
import pandas as pd
from math import floor
import scipy.misc
import random
from imageio import imread
import read_kneeDataset as kd

"""
Parameters to split data into groups for training, validation, and testing  

src_img:            directory containing the set of low quality or high quality images 
split_proportion:   proportion of data into the training, validation, and testing groups respectively
"""

def get_train_val_test(proportion):
    scans = kd.get_unique_scans()
    random.shuffle(scans)

    training = scans[0:floor(len(scans) * proportion[0])]
    validation = scans[floor(len(scans) * proportion[0]):floor(len(scans) * proportion[0]) + floor(len(scans) * proportion[1])]
    test = scans[floor(len(scans) * proportion[0]) + floor(len(scans) * proportion[1]):len(scans)]

    return training, validation, test

def create_csv_files(proportion):

    if not os.path.isfile('train_knee.csv') and not os.path.isfile('validation_knee.csv') and not os.path.isfile('test_knee.csv'):
        train_file = open('train_knee.csv','w')
        val_file = open('validation_knee.csv','w')
        test_file = open('test_knee.csv','w')

        train_set, val_set, test_set= get_train_val_test(proportion)

        for z in train_set:
            for s in kd.get_slices_from_scan(z):
                train_file.write('{}, {}\n'.format(z,s))

        for z in val_set:
            for s in kd.get_slices_from_scan(z):
                val_file.write('{}, {}\n'.format(z,s))

        for z in test_set:
            for s in kd.get_slices_from_scan(z):
                test_file.write('{}, {}\n'.format(z,s))

        train_file.close()
        val_file.close()
        test_file.close()
    else:
        print('Data already splitted into training, validation, and testing')

def data_mean_value(csv, nmax):


    data = pd.read_csv(csv)
    r, c = data.shape
    samples_mean = np.zeros(r)
    idSamples = 0
    for i , b in data.head(nmax).iterrows():
        slices = kd.get_slices_from_scan(b[0])
        slices_mean = np.zeros(len(slices))
        idSlices = 0
        for s in slices:
            slices_mean[idSlices] = np.mean(kd.getMRISlice(b[0], s), axis=(0,1))
            idSlices += 1
        samples_mean[idSamples] = np.mean(slices_mean, axis=0)
        idSamples += 1

    return np.mean(samples_mean,axis=0)



if __name__ == "__main__":

   #print(data_mean_value('test_knee.csv'))
   #data_mean_value('test_knee.csv')
   create_csv_files([0.45,0.05,0.5])



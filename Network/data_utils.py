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


"""
Parameters to split data into groups for training, validation, and testing  

src_img:            directory containing the set of low quality or high quality images 
split_proportion:   proportion of data into the training, validation, and testing groups respectively
"""



def get_unique_scans(src_data):

    scans = []
    for file in os.listdir(src_data):
        temp = file.split("_")
        new_scan = temp[0]
        if new_scan not in scans:
            scans.append(new_scan)
    return scans

def get_slices_from_scan(src_data, scan):

    slices = []
    for file in os.listdir(src_data):
        if file.startswith(scan):
            slices.append(file)

    return slices

def get_train_val_test(src_data, proportion):

    scans = get_unique_scans(src_data)
    random.shuffle(scans)

    training = scans[0:floor(len(scans) * proportion[0])]
    validation = scans[floor(len(scans) * proportion[0]):floor(len(scans) * proportion[0]) + floor(len(scans) * proportion[1])]
    test = scans[floor(len(scans) * proportion[0]) + floor(len(scans) * proportion[1]):len(scans)]

    data_training = []
    data_validation = []
    data_test = []

    for i in training:
        data_training.extend(get_slices_from_scan(src_data, i))

    for j in validation:
        data_validation.extend(get_slices_from_scan(src_data, j))

    for z in test:
        data_test.extend(get_slices_from_scan(src_data, z))


    return data_training, data_validation, data_test

def create_csv_files(src_data, proportion):

    if not os.path.isfile('train_apples.csv') and not os.path.isfile('validation_apples.csv') and not os.path.isfile('test_apples.csv'):
        train_file = open('train_apples.csv','w')
        val_file = open('validation_apples.csv','w')
        test_file = open('test_apples.csv','w')

        train_set, val_set, test_set= get_train_val_test(src_img, proportion)

        for z in train_set:
            train_file.write(z + ',' + z[0:len(z)-3]+ 'npy' + '\n')

        for z in val_set:
            val_file.write(z + ',' + z[0:len(z)-3]+ 'npy' + '\n')

        for z in test_set:
            test_file.write(z + ',' + z[0:len(z)-3]+ 'npy' + '\n')

        train_file.close()
        val_file.close()
        test_file.close()
    else:
        print('Data already splitted into training, validation, and testing')

def data_mean_value(csv, dir):
    data = pd.read_csv(csv)
    r, c = data.shape
    values = np.zeros(r)
    idx = 0
    for i, row in data.iterrows():
        img = imread(dir+row[0], pilmode="F")
        values[idx] = np.mean(img, axis=(0,1))
        idx += 1

    return np.mean(values,axis=0)



if __name__ == "__main__":
    src_img = "/home/luis/Desktop/Datasets/Apple CT/recs-10-projs/input/"
    split_proportion = [.8, .1, .1]
    #print(get_unique_scans(src_img))
    create_csv_files(src_img, split_proportion)


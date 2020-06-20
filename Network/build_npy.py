# title                 :build_npy.py
# description           :It converts label images into .npy format
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
from scipy import misc
import matplotlib.pyplot as plt
from imageio import imread

"""
Parameters for converting output images into .npy files

high_quality_dir:   directory of the target images 
target_dir:         directory where the .npy files will be saved
files_ext:          extension of images files at low_quality_dir and high_quality_dir
debug:              flag to allow intermediate visualization

"""

high_quality_dir = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_discrete/'
target_dir = "/home/luis/Desktop/Datasets/Apple CT/recs-10-projs/output/"

files_ext = '.png'

debug = False

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

for file in os.listdir(high_quality_dir):

        if file.endswith(files_ext):

            output_img = imread(os.path.join(high_quality_dir,file), pilmode='F')
            values = np.unique(output_img)
            h, w = output_img.shape
            gt = np.zeros((len(values), h,w))

            for i in range(len(values)):
                gt[i][output_img == values[i]] = 1

            if debug:
                print(np.unique(output_img))
                print('min: {} max: {}'.format(np.min(output_img), np.max(output_img)))
                plt.figure()
                plt.imshow(gt[4,:,:], cmap='gray')
                plt.show()
                break
            else:

                np.save(target_dir+file[0:len(file)-3]+'npy', gt)
                print(target_dir+file+ " Done!")




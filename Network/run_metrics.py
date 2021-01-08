
from matplotlib import pyplot as plt
import read_kneeDataset as kd
import numpy as np
import imageio as io
import os
import re


if __name__ == '__main__':

    src= '/home/antwerpdnn/PycharmProjects/code-sprint-2020/Network/results-Network-FCN-8-2nd-OAI-dataset/'

    all_files = os.listdir(src)
    size = len(all_files)
    all_types = ['TB', 'FB', 'FC', 'TC']


    dices = {'TB': [], 'FB': [], 'FC': [], 'TC': []}

    for file_idx in range(size):

        num = re.findall(r'\d+', all_files[file_idx])
        code = num[0]
        slice = int(num[1])


        if slice > 0 and slice < 160:
            for type in all_types:
                prediction = np.load(src + '{}_slice_{}.npy'.format(code, slice))
                gt = kd.getSegmentationSlice(kd.getSegmentationCode(code), slice-1)
                if type == 'TB':
                    gt = gt == 3
                    prediction = prediction == 3
                if type == 'FB':
                    gt = gt == 1
                    prediction = prediction == 1
                if type == 'FC':
                    gt = gt == 2
                    prediction = prediction == 2
                if type == 'TC':
                    gt = gt == 4
                    prediction = prediction == 4
                if np.sum(gt) > 0:
                    intersec = np.logical_and(gt,prediction)
                    val = 2*np.sum(intersec)/(np.sum(gt)+np.sum(prediction))
                    #if len(np.unique(prediction)) < 5 and np.sum(gt) > 0:
                    #    print("type: {} code: {} slice: {}".format(type, code, slice))

                    dices[type].append(val)
                    if (type == 'FB' or type =='TB') and val < 0.90:
                        print("\n\nBANG!")
                        print("code {} slice {} type {} val {}!!\n\n".format(code, slice, type, val))
                    if (type == 'TP') and val < 0.80:
                        print("\n\nBANG!")
                        print("code {} slice {} type {} val {}!!\n\n".format(code, slice, type, val))
                    if (type == 'TC') and val < 0.70:
                        print("\n\nBANG!")
                        print("code {} slice {} type {} val {}!!\n\n".format(code, slice, type, val))

        #print(file_idx/size)


print('AVERAGE DICES:')
#print(dices['TB'])
print('TB: {}'.format(np.mean(dices['TB'])))
print('FB: {}'.format(np.mean(dices['FB'])))
print('FC: {}'.format(np.mean(dices['FC'])))
print('TC: {}'.format(np.mean(dices['TC'])))
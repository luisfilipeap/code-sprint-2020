import SimpleITK as sitk
import numpy as np
from PIL import Image as image
import pydicom
from matplotlib import pyplot as plt
import os

def get_unique_scans():
    src = '/home/antwerpdnn/Documents/TrainedNetworkJonas/TrainingData/'
    scans = []
    for folder in os.listdir(src):
        if os.path.isdir(src+folder):
            scans.append(folder)

    return scans

def get_slices_from_scan(scan):
    src = '/home/antwerpdnn/Documents/TrainedNetworkJonas/TrainingData/{}/'.format(scan)
    slices = []
    for file in os.listdir(src):
            slices.append(file)

    return slices

def getSegmentationCode(MRIcode):
    map_file_src = '/home/antwerpdnn/Documents/TrainedNetworkJonas/OAI-ZIB/doc/oai_mri_paths.txt'
    map_file = open(map_file_src,'r')
    for line in map_file:
        parts = line.split('/')
        if int(MRIcode) == int(parts[3][0:-1]):
            return parts[1]
    return -1

def getMRISlice(code, slice):
    dcm_file = '/home/antwerpdnn/Documents/TrainedNetworkJonas/TrainingData/{}/{}'.format(code, str(slice).zfill(3))
    dcm = pydicom.dcmread(dcm_file)
    dcm = dcm.pixel_array
    dcm = np.flip(dcm, axis=0)
    dcm[dcm > 255] = 255
    dcm = dcm / 255.
    return dcm

def getSegmentationSlice(code, slice):
    full_mhd_path = "/home/antwerpdnn/Documents/TrainedNetworkJonas/OAI-ZIB/segmentation_masks/{}.segmentation_masks.mhd".format(code)
    itkimage = sitk.ReadImage(full_mhd_path)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    img = image.fromarray(numpyImage[:,:,slice], 'L')
    return np.array(img)



if __name__ == '__main__':
    c = '10457612'
    slice = 160
    s = getMRISlice(c,slice)
    #print(s.shape)
    plt.figure('original')
    plt.imshow(s)
    plt.figure('segmentation')
    g = getSegmentationSlice(getSegmentationCode(c),slice)
    #print(g.shape)
    #print(np.unique(g))
    plt.imshow(g)
    plt.show()
    import imageio
    imageio.imwrite('/home/antwerpdnn/Documents/TrainedNetworkJonas/Final-Files/{}_{}_seg.png'.format(c,slice), g*60)
    imageio.imwrite('/home/antwerpdnn/Documents/TrainedNetworkJonas/Final-Files/{}_{}_mri.png'.format(c, slice), s)

    '''src = '/home/antwerpdnn/Documents/TrainedNetworkJonas/TrainingData/'
    for folder in os.listdir(src):
        if os.path.isdir(src+folder):
            print("MRI: {} SEG: {}".format(folder, getSegmentationCode(folder)))'''


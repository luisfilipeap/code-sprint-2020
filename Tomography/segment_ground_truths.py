import os
from imageio import imread, imwrite
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


src = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_png/'
dest = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_discrete/'


if not os.path.isdir(dest):
    os.mkdir(dest)

for file in os.listdir(src):
    i = imread(src+file)
    i = (i-np.min(i))/(np.max(i)-np.min(i))

    t = np.linspace(0,1,6)
    final = np.digitize(i, bins=t)
    print(np.unique(final))

    #imwrite(dest+file,final)


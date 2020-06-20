import os
from imageio import imread, imwrite
import numpy as np

src = '/home/luis/Desktop/Datasets/Apple CT/ground_truths/'
dest = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_png/'

if not os.path.isdir(dest):
    os.mkdir(dest)

for file in os.listdir(src):
    i = imread(src+file)
    i = (i-np.min(i))/(np.max(i)-np.min(i))
    imwrite(dest+file[0:len(file)-3]+'png',i)
    print(file)



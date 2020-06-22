
import matplotlib.pyplot as plt
from imageio import imread

file = '31201_181.png'

src_output = '/home/luis/PycharmProjects/public-code-sprint-2020/Network/results-Network-SOTA-10-projs/'
src_input  = '/home/luis/Desktop/Datasets/Apple CT/recs-10-projs/input/'
src_gt  = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_discrete/'


output = imread(src_output+file)
input  = imread(src_input+file)
gt     = imread(src_gt+file)

plt.figure('OUTPUT')
plt.imshow(output)
plt.figure('INPUT')
plt.imshow(input)
plt.figure('GT')
plt.imshow(gt)
plt.show()
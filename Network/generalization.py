import torch
import time
import os
from torch.autograd import Variable
import scipy.misc
import scipy
from Network.data_loader import Tomographic_Dataset
from torch.utils.data import Dataset, DataLoader
from Network.data_utils import data_mean_value
import numpy as np
import ntpath
from matplotlib import pyplot as plt
from torchvision import utils
from imageio import imwrite

net             = "Network-SOTA"
n_class         = 6
projs           = 10
input_dir       = "/home/luis/Desktop/Datasets/Apple CT/recs-{}-projs/input/".format(projs)
target_dir      = "/home/luis/Desktop/Datasets/Apple CT/recs-{}-projs/output/".format(projs)
means           = data_mean_value("train_apples.csv", input_dir) / 255.
print(means)

model_src = "./models/{}-{}-projs-model-apple-dataset".format(net, projs)



def evaluate_img():

    test_data = Tomographic_Dataset(csv_file="test_apples.csv", phase='val', flip_rate=0, input_dir=input_dir, target_dir=target_dir, n_class=6, train_csv="train_apples.csv")
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

    fcn_model = torch.load(model_src)
    n_tests = len(test_data.data)

    print("{} files for testing....".format(n_tests))

    folder = "./results-{}-{}-projs/".format(net, projs)
    if not os.path.exists(folder):
        os.makedirs(folder)


    count = 0
    for iter, batch in enumerate(test_loader):

        name = batch['file'][0]

        input = Variable(batch['X'].cuda())

        output = fcn_model(input)

        count = count + 1
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        pred = pred[0,:,:]
        imwrite(folder+'/'+name, pred)









        #print("executed {} of {}\n".format(iter,len(test_loader)))

    #print("mean: {}".format(np.mean(execution_time[1:n_tests])))
    #print("std: {}".format(np.std(execution_time[1:n_tests])))



if __name__ == "__main__":
    evaluate_img()


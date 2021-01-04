# -*- coding: utf-8 -*-

from __future__ import print_function
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Network.data_loader import Tomographic_Dataset
from Network.FCN import VGGNet, FCN

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import misc
import os



net = "Network-FCN-8"

batch_size = 10 #antes 10
epochs     = 100

momentum   = 0.5
w_decay    = 1e-5 #antes 1e-5
#after each 'step_size' epochs, the 'lr' is reduced by 'gama'
lr         = 0.001 # reference le-4
step_size  = 100
gamma      = 0.5

n_class = 6

configs         = "{}-model-OAI-dataset".format(net)
train_file      = "train_knee.csv"
val_file        = "validation_apples.csv"


validation_accuracy = np.zeros((epochs,1))

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, configs)
use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
print("GPU Available: ",use_gpu, " number: ",len(num_gpu))

train_data = Tomographic_Dataset(csv_file=train_file, phase='train', train_csv=train_file, n_class=5)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

#directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
val_data = Tomographic_Dataset(csv_file=val_file, phase='val', flip_rate=0, train_csv=train_file, n_class=5)
val_loader = DataLoader(val_data, batch_size=1, num_workers=6)


vgg_model = VGGNet(pretrained=False, requires_grad=True, remove_fc=True)
fcn_model = FCN(pretrained_net=vgg_model, n_class=6)


if use_gpu:
    ts = time.time()
    if net.startswith('VGG-UNET'):
        vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)



def train():
    hit = 0
    delta = 0.000001
    for epoch in range(epochs):
        scheduler.step()
        if epoch > 2 and abs(validation_accuracy[epoch-2]-validation_accuracy[epoch-1]) < delta:
            hit = hit + 1
        else:
            hit = 0
        if hit == 5:
            break

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def val(epoch):
    fcn_model.eval()
    pixel_acc_list = []

    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()
        target = batch['Y'].data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        target = target.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        for p, t in zip(pred, target):
            pixel_acc_list.append(pixel_acc(p, t))


    accs = np.mean(pixel_acc_list)
    validation_accuracy[epoch] = accs


    print("epoch{}, mse_acc: {}".format(epoch,accs))


def pixel_acc(pred, target):

    correct = (pred == target).sum()
    total   = (target == target).sum()

    return correct/total

if __name__ == "__main__":
    #val(0)  # show the accuracy before training
    start = time.time()
    train()
    end = time.time()
    duration = end - start

    d = datetime(1, 1, 1) + timedelta(seconds=int(duration))
    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day - 1, d.hour, d.minute, d.second))

    np.save('validation_accuracy_{}-apple-dataset.npy'.format(net), validation_accuracy)

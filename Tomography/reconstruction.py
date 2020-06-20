#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:13:17 2020

@author: danfre
"""

import astra
from imageio import imread, imwrite
import odl
from Tomography.OpTomo import OpTomo
from Tomography.OpTomoDART_module import *
import os

# data path
in_path = '/home/luis/Desktop/Datasets/Apple CT/ground_truths_png/'
out_path = '/home/luis/Desktop/Datasets/Apple CT/recs-10-projs/'

if not os.path.isdir(out_path):
    os.mkdir(out_path)

out_path = out_path+'input/'

if not os.path.isdir(out_path):
    os.mkdir(out_path)

seq = os.listdir(in_path)
count = 0
total = len(seq)
for file in seq:

    data = imread(in_path+file, pilmode='F')/256.

    det_size   = data.shape[1]
    n_angles   = 10
    vol_size   = [256,256]
    part = odl.uniform_partition(0, np.pi, n_angles)

    angles = part.grid.coord_vectors
    proj_geom  = astra.create_proj_geom('parallel',1.0,det_size,angles)
    vol_geom   = astra.create_vol_geom(vol_size[0],vol_size[1])
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

    W                = OpTomo(proj_id)
    _, p         = astra.create_sino(data, proj_id)
    p                = p.ravel()

    rec = doSirt(W,p,100)

    imwrite(out_path+file, np.reshape(rec*255,vol_size))
    print("{}".format(count/total))
    count = count + 1




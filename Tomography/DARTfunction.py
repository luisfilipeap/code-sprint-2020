# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:30:17 2020

@author: danfr
"""
import astra
import numpy as np
import scipy.sparse.linalg
import scipy.io
from scipy.sparse.linalg import lsqr
from Tomography.OpTomo import OpTomo
import pylab
import pylops
from scipy.io import savemat
import torch
from imageio import imread
import matplotlib.pyplot as plt



def doDART(W,p,Params):
    gray_values = Params['gray_values']
    t0          = Params['t0']
    t           = Params['t']
    max_iter    = Params['max_iter']
    p_val       = Params['p_val']
    sz          = Params['sz']
    b           = Params['b']
    
    # INITIAL SIRT STEPS
    rec = doSirt(W, p, t0)
    for k in range(max_iter):
        s         = segmentImage(rec)
        mask      = getMask(s,sz,gray_values,p_val)
        rec       = doMaskedSirt(W,p,t,mask,mask*rec + np.logical_not(mask)*s)
#        rec       = doSmoothing(rec,sz,b)
    return segmentImage(rec,tl,tu,gray_values)

def find_Boundary(im):
    gray_values = np.unique(im)
    # This function uses a element wise implementation of the uniform kernel to calculate the boundary.
    # im: The image to calculate boundary for
    # gray_values: an array of the different gray_values in the image.
    # RETURN: np.array containing the number of different neighbours each pixel has.
    freedom_matrix = -np.ones(im.shape,dtype='uint8')
    b = np.zeros(im.shape,dtype= 'uint8')
    c = np.zeros(im.shape,dtype= 'uint8')    
    if len(im.shape) == 3:
        d = np.zeros(im.shape,dtype= 'uint8')    
        for gv in gray_values:
            temp = (im == gv)
            A = shiftSum3D(np.logical_not(temp).astype('uint8'),b,c,d)
            freedom_matrix[temp] = freedom_matrix[temp] + A[temp] + 1
        freedom_matrix[freedom_matrix == -1] = 8
    else:
        for gv in gray_values:
            temp = (im == gv)
            A = shiftSum(np.logical_not(temp).astype('uint8'),b,c)
            freedom_matrix[temp] = freedom_matrix[temp] + A[temp] + 1
        freedom_matrix[freedom_matrix == -1] = 8
    return freedom_matrix

def getMask(im,sz,gray_values,p,minNeighbours = 0):
    B = find_Boundary(np.reshape(im,sz)) > minNeighbours
    P = np.random.random(B.shape) < p
    return np.ravel(np.logical_or(B,P))

def doSirt(W,p,n_iter):
    rec     = np.zeros(W.shape[1])
    rowSums = W._matvec(np.ones(W.shape[1]))
    rowSums[rowSums < 0.000001] = np.inf
    np.divide(1,rowSums,out = rowSums)
    
    colSums = W._rmatvec(np.ones(W.shape[0]))
    colSums[colSums < 0.000001] = np.inf
    np.divide(1,colSums,out = colSums)
    for k in range(n_iter):
        np.add(rec,colSums * ( W._rmatvec((rowSums*(p-W._matvec(rec))))),out = rec)
        rec[rec < 0] = 0
    return rec

def doMaskedSirt(W,p,n_iter,mask = 0,s = 0):
    if type(mask) == int:
        if type(s) == int:
            return doSirt(W,p,n_iter)
    else:
        mask = np.ravel(mask)
        
    if type(s) == int:
        s = np.zeros(W.shape[1])
    p_res = p - W._matvec((s*np.logical_not(mask)))
    rowSums = W._matvec(np.ones(W.shape[1]) * mask)
    rowSums[rowSums < 0.000001] = np.inf
    np.divide(1,rowSums,out = rowSums)
    
    colSums = (W._rmatvec(np.ones(W.shape[0]))) * mask
    colSums[colSums < 0.000001] = np.inf
    np.divide(1,colSums,out = colSums)
    
    rec = s.copy()
    for k in range(n_iter):
        # print(':::SIRT iteration %d:::' %(k))
        np.add(rec,colSums * ( W._rmatvec((rowSums*(p_res-W._matvec(rec*mask))))),out = rec)
        rec[rec < 0] = 0
    np.add(rec*mask,s*np.logical_not(mask),out=rec)
    return rec


def shiftSum(a,b,c):
    b[:,0] = a[:,0]
    b[:,1:] = a[:,1:] + a[:,:-1]
    b[:,:-1] = b[:,:-1] + a[:,1:]
    c[0,:] = b[0,:]
    c[1:,:] = b[1:,:] + b[:-1,:]
    c[:-1,:] = c[:-1,:] + b[1:,:]
    return c
def shiftsumMemory(a,b):
    b[:,1:]   = a[:,1:]
    a[:,1:]  += a[:,:-1]
    a[:,:-1] += b[:,1:]
    b[1:,:]   = a[1:,:]
    a[1:,:]  +=  a[:-1,:]
    a[:-1,:] +=  b[1:,:]
    return a
def shiftSum3D(a,b,c,d):
    b[:,:,0] = a[:,:,0]
    b[:,:,1:] = a[:,:,1:] + a[:,:,:-1]
    b[:,:,:-1] = b[:,:,:-1] + a[:,:,1:]
    c[:,0,:] = b[:,0,:]
    c[:,1:,:] = b[:,1:,:] + b[:,:-1,:]
    c[:,:-1,:] = c[:,:-1,:] + b[:,1:,:]
    d[0,:,:]   = c[0,:,:]
    d[1:,:,:]  = c[1:,:,:] + c[:-1,:,:]
    d[:-1,:,:] = d[:-1,:,:] + c[1:,:,:]
    return d
    

def segmentImage(rec):
    # The Network should be called here
    n_class = 6
    model_src = "../Network/models/Network-SOTA-10-projs-model-apple-dataset"
    fcn_model = torch.load(model_src)

    rec = rec - 0.3102145734091077
    rec_torch = torch.from_numpy(rec.copy()).float()
    data_in = torch.zeros(1, 1, 256, 256)
    data_in[0, 0, :, :] = rec_torch

    output = fcn_model(data_in)
    output = output.data.cpu().numpy()
    N, _, h, w = output.shape
    pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
    pred = pred[0, :, :]
    return pred

if __name__ == "__main__":
    #JUST TESTING SEGMENTATION
    i = imread('/home/luis/Desktop/Datasets/Apple CT/recs-10-projs/input/32206_427.png')
    seg_i = segmentImage(i)

    plt.imshow(seg_i)
    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:28:03 2020

@author: danfre
"""

import numpy as np
import pylops

def aid(x):
    # This function returns the memory
    # block address of an array.
    return x.__array_interface__['data'][0]

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

def doMaskedLambdaSirt(W,p,n_iter,mask = 0,s = 0):
    if type(mask) == int:
        if type(s) == int:
            return doSirt(W,p,n_iter)
    else:
        mask = np.ravel(mask)
    alpha = np.sum(mask != 0)/float(np.prod(mask.shape))
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
        np.add(rec,alpha*colSums * ( W._rmatvec((rowSums*(p_res-W._matvec(rec*mask))))),out = rec)
        rec[rec < 0] = 0
    np.add(rec*mask,s*np.logical_not(mask),out=rec)
    return rec

def getThresholds(gray_values):
    gray_values  = gray_values
    tl = [0.]
    tu = []
    distance = []
    not_init = True
    for k in range(len(gray_values)-1):
        th = (gray_values[k+1] + gray_values[k])*0.5
        tl.append(th) 
        tu.append(th)
    tu.append(1)
    return tl,tu

def segmentImage(rec,tl,tu,gray_values):
    seg = np.copy(rec)
    seg = seg.astype(float)
    for k in range(len(tl)):
       nk = (rec >= tl[k])
       mk = (rec < tu[k])
       seg[np.logical_and(nk,mk)] = gray_values[k]
    seg[seg < tl[0]]  = gray_values[0]
    seg[seg > tu[-1]] = gray_values[-1]
    return seg


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
    
def find_Boundary(im,gray_values):
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



def apply2D(V_in,b):
    center         = np.zeros([3,3])
    center[1,1]    = 1
    average        = np.ones([3,3])/8.
    average[1,1]   = -1
    kernel         = center + b*average
    r              = 1
    w              = 2 * r + 1;
    V_out          = np.zeros([V_in.shape[0] + w-1, V_in.shape[1] + w-1])
    for s in np.arange(-r,r+1,1):
        for t in np.arange(-r,r+1,1):
         V_out[(r+s):(V_out.shape[0]-r+s),(r+t):(V_out.shape[1]-r+t)] = V_out[(r+s):(V_out.shape[0]-r+s),(r+t):(V_out.shape[1]-r+t)] + kernel[r+s,r+t]*V_in
    V_out = V_out[r:-r,r:-r]
    return V_out

def apply3D(V_in,b):
    center         = np.zeros([3,3,3])
    center[1,1,1]  = 1
    average        = np.ones([3,3,3])/26
    average[1,1,1] = -1
    kernel         = center + b*average
    r              = 1
    w              = 2 * r + 1;
    V_out          = np.zeros([V_in.shape[0] + w-1, V_in.shape[1] + w-1,V_in.shape[2] + w-1])
    for s in np.arange(-r,r+1,1):
        for t in np.arange(-r,r+1,1):
            for u in np.arange(-r,r+1,1):
                V_out[(r+s):(V_out.shape[0]-r+s),(r+t):(V_out.shape[1]-r+t),(r+u):(V_out.shape[2]-r+u)] = V_out[(r+s):(V_out.shape[0]-r+s),(r+t):(V_out.shape[1]-r+t),(r+u):(V_out.shape[2]-r+u)] + kernel[r+s,r+t,r+u]*V_in
    V_out = V_out[r:-r,r:-r,r:-r]
    return V_out        

def doSmoothing(V_in,size,b):
    V_in = np.reshape(V_in,size)
    if (abs(b) < 1e-16):
        return np.ravel(V_in)
    if len(V_in.shape) == 3:
        return np.ravel(apply3D(V_in,b))
    else:
        return np.ravel(apply2D(V_in,b))

def getMask(im,sz,gray_values,p,minNeighbours = 0):
    B = find_Boundary(np.reshape(im,sz),gray_values) > minNeighbours
    P = np.random.random(B.shape) < p
    return np.ravel(np.logical_or(B,P))

def doDART(W,p,Params):
    gray_values = Params['gray_values']
    t0          = Params['t0']
    t           = Params['t']
    max_iter    = Params['max_iter']
    p_val       = Params['p_val']
    sz          = Params['sz']
    b           = Params['b']
    # print(p_val)
    # initmask    = np.ravel(Params['mask'])
    initmask = 0
    GT          = np.ravel(Params['GT'])
    errors      = np.zeros(max_iter)
    # CALCULATE THRESHOLDS
    tl,tu = getThresholds(gray_values)
    
    # INITIAL SIRT STEPS
    rec = doMaskedSirt(W, p, t0,initmask)
    s   = np.zeros(rec.shape)
    for k in range(max_iter):
        s         = segmentImage(rec,tl,tu,gray_values)
        errors[k] = np.sum(s != GT)
        mask      = getMask(s,sz,gray_values,p_val)
        # errors[k] = np.sum(mask == 1)
        rec       = doMaskedSirt(W,p,t,mask,mask*rec + np.logical_not(mask)*s)
        rec       = doSmoothing(rec,sz,b)
        # print("::: DART iteration %d Finished, error = %d :::" %(k,errors[k]))
    return segmentImage(rec,tl,tu,gray_values), errors    

def doADART(W,p,Params):
    gray_values = Params['gray_values']
    t0          = Params['t0']
    t           = Params['t']
    max_iter    = Params['max_iter']
    p_val       = Params['p_val']
    sz          = Params['sz']
    b           = Params['b']
    print(p_val)
    # initmask    = np.ravel(Params['mask'])
    initmask = 0
    GT          = np.ravel(Params['GT'])
    errors      = np.zeros(max_iter)
    # CALCULATE THRESHOLDS
    tl,tu      = getThresholds(gray_values)
    iter_chunk = int(np.ceil(max_iter/4.))
    # INITIAL SIRT STEPS
    rec = doMaskedSirt(W, p, t0,initmask)
    s   = np.zeros(rec.shape)
    minNeighbours = 0
    for its in range(4):
        for k in range(iter_chunk):
            s         = segmentImage(rec,tl,tu,gray_values)
            errors[minNeighbours*iter_chunk + k] = np.sum(s != GT)
            mask      = getMask(s,sz,gray_values,p_val,minNeighbours)
            rec       = doMaskedSirt(W,p,t,mask,mask*rec + np.logical_not(mask)*s)
            rec       = doSmoothing(rec,sz,b)
            # print("::: DART iteration %d Finished, error = %d :::" %(k,errors[k]))
        minNeighbours += 1
    return segmentImage(rec,tl,tu,gray_values), errors  

def updateMap(s,rec,size,probabilityMap,mask,gray_values):
    tl,tu  = getThresholds(gray_values)
    s_next = segmentImage(rec, tl, tu, gray_values)
    change = s != s_next
    probabilityMap[mask] /= 2
    probabilityMap[change] = 1
    del change
    probabilityMap[np.ravel(find_Boundary(np.reshape(s,size),gray_values)) > 0] = 1
    return probabilityMap.ravel(),s_next
    
def getMaskAU(probabilityMap):
    return np.ravel(np.random.random(probabilityMap.shape) < probabilityMap)

def doAUDART(W,p,Params):
    gray_values    = Params['gray_values']
    t0             = Params['t0']
    t              = Params['t']
    max_iter       = Params['max_iter']
    p_val          = Params['p_val']
    sz             = Params['sz']
    b              = Params['b']
    mask           = 0
    # probabilityMap = np.ravel(Params['initial'])
    GT             = np.ravel(Params['GT'])
    errors         = np.zeros(max_iter)
    # CALCULATE THRESHOLDS
    tl,tu = getThresholds(gray_values)
    
    # INITIAL SIRT STEPS
    rec            = doMaskedSirt(W, p, t0,mask)
    temp           = IWD(np.reshape(rec,sz),gray_values)
    probabilityMap = np.ravel(calcEntropy(temp,3))
    del temp
    for k in range(max_iter):
        s                = segmentImage(rec,tl,tu,gray_values)
        errors[k]        = np.sum(s != GT)
        mask             = getMaskAU(probabilityMap)
        # errors[k] = np.sum(mask == 1)
        rec              = doMaskedSirt(W,p,t,mask,mask*rec + np.logical_not(mask)*s)
        rec              = doSmoothing(rec,sz,b)
        probabilityMap,s = updateMap(s, rec, sz,probabilityMap,mask, gray_values)
        # print("::: AU-DART iteration %d Finished, error = %d :::" %(k,errors[k]))
    return s,errors

def calcEntropy(p,n = 0):
    H = np.zeros(p.shape)
    temp = p > 0
    if n == 0:
        H[temp] = p[temp]*np.log(p[temp])/np.log(p.shape[0])
    else:
        H[temp] = p[temp]*np.log(p[temp])/np.log(2)
    return -np.sum(H,0)

def IWD(img,gray_values):
    if len(img.shape) == 2:
        return IWD_2D(img,gray_values)
    else:
        return IWD_3D(img,gray_values)

def IWD_2D(img,gray_values):
    #Calculates the inverse weighed distance to create a numpy array of probabilities for calculating information entropy
    a = np.zeros([len(gray_values),img.shape[0],img.shape[1]])
    for k in range(a.shape[0]):
        a[k,:,:] = np.abs(img - gray_values[k])
    a[a < 0.01] = 0.01
    a = 1/a
    a = a/(np.sum(a,0))
    return a

def IWD_3D(img,gray_values):
    a = np.zeros([len(gray_values)] + list(img.shape))
    for k in range(a.shape[0]):
        a[k,:,:,:] = np.abs(img - gray_values[k])
    a[a < 0.01] = 0.01
    a = 1/a
    a = a/(np.sum(a,0))
    return a

def getDerivativeOperators(imageSize):
    Dop = [pylops.FirstDerivative(imageSize[0]*imageSize[1],dims=imageSize,dir=0),pylops.FirstDerivative(imageSize[0]*imageSize[1],dims=imageSize,dir=1)]
    return Dop

def doTVMin(W,p,Params):
    mu       = Params['mu']
    lam      = Params['lam']
    max_iter = Params['max_iter']
    t        = Params['t']
    sz       = Params['sz']
    Dop      = getDerivativeOperators(sz)
    xinv, niter = \
    pylops.optimization.sparsity.SplitBregman(W, Dop, p.flatten(),
                                              max_iter, t,
                                              mu=mu, epsRL1s=lam,
                                              tol=1e-4, tau=1., show=True,
                                              **dict(iter_lim=5, damp=1e-4))
    return xinv

def calculateRRE(W,p,s,n_iter):
    ps = p - W*s
    e  = doSirt(W,ps,n_iter)
    return e
def DART_RRE(W,p,e_previous,s,n_iter,mask):
    ps   = p - W*s
    e    = doMaskedSirt(W,p,n_iter,mask,e_previous)
    return e    
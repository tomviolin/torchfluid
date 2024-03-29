#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:23:55 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
parser = argparse.ArgumentParser(description='torchfluid.py')
parser.add_argument('-n', '--npy', action='store_true', help='convert raw lbinfo input to npy files')
parser.add_argument('-e', '--new-evo', action='store_true', help='create new evo')
parser.add_argument('-k', '--kdsize', type=int, default=1, help='size of kernel used to chop up the flow field for training')
parser.add_argument('-r', '--resume', type=str, default=None, help='resume the evo model')
parser.add_argument('-d', '--debug', action='store_true', help='debug')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
parser.add_argument('-c', '--cuda', action='store_true', help='cuda')
parser.add_argument('-q', '--quiet', action='store_true', help='quiet')
parser.add_argument('-u', '--unittest', action='store_true', help='unittest')
parser.add_argument('-l', '--loadrecs', type=int,default=6000, help='number of recs to load')

import copy
import gc
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime as dt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_dct as dct
from glob2 import glob
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist
from torch import optim
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
#from tqdm import tqdm

ITERS = 1 # number of iterations
NPYDIR = 'npydata' # npydata directory
EVODIR = 'evo' # evo directory
KDSIZE = 1 # kernel size

TIMESTAMP = None # timestamp
RESUME = None # resume
SAVE = None # save
LOAD = None # load
DEBUG = False # debug
VERBOSE = False # verbose
PLOT = False # plot
CUDA = False # cuda
GPU = 0 # gpu
MODEL = None # model
AGENT = None # agent
BATCHSIZE = None # batchsize
FILE = None # file
OUTPUT = None # output
EXECUTE = None # execute
WRITE = None # write
YAML = None # yaml
ZIP = None # zip
JSON = None # json
QUIET = False # quiet
UNITTEST = False # unittest

# parse command line arguments
args = parser.parse_args()
if args.kdsize: KDSIZE = args.kdsize
if args.resume: RESUME = args.resume
VERBOSE = args.verbose
DEBUG = args.debug
def get_timestamp():
    """get timestamp"""
    return dt.now().strftime("%Y%m%d_%H%M%S.%f")

def get_program_timestamp():
    """get program timestamp"""
    return program_timestamp

program_timestamp = get_timestamp()

def childname(pn):
    # this is used to create a unique name for the child parameter

    cn = pn
    if cn[-1].isnumeric():
        cn=cn+"_000001c"
        return cn
    sn = int(cn[-7:-1])+1
    sn=f"{sn:06d}"
    cn=cn[:-7]+sn+cn[-1]
    return cn

def mp(MPDICT,pn,dflt=None,force=None):
    # this function is used to get or set a parameter value
    pns = pn.split(":")
    for i in range(len(pns)):
        if pns[i].isnumeric():
            pass
        else:
            pns[i]='"'+pns[i]+'"'

    pnstr = "MPDICT["+"][".join(pns)+"]"
    if force is not None:
        if isinstance(force,str):
            expr = pnstr+f" = '{force}'"
        else:
            expr = pnstr+f" = {np.around(force,6)}"
        if DEBUG: print(f"EXEC: {expr}")
        exec(expr)
        return force

    try:
        #print(f"{pnstr}",end='')
        paramval = eval(pnstr)
        #print(f">>{paramval}<< ",end='',flush=True)
        if isinstance(paramval,list) and not isinstance(paramval[0],list) and not isinstance(paramval[0],dict):
            #print("[rndizer]",end='')
            pmin=paramval[0]
            pmax=paramval[1]
            pint=paramval[2]
            numint = np.floor((pmax-pmin)/pint) + 1
            intv = np.random.randint(0,numint)
            paramval = pmin + pint*intv
            exec(pnstr + f" = {np.around(paramval,6)}")
        return paramval
    except Exception as e:
        print(f"PARAM WARNING: {e}: "+pnstr)
        return dflt

def lbinfodatagenerator():
    # this function is used to generate data from the lbinfo model
    # read last file
    fname = sorted(glob("/home/tomh/lbinfo*.csv"))[-1]
    print(f"fname={fname}")
    fbase = fname.split("/")[-1].split(".")[-2]
    os.makedirs(f"data/{fbase}",exist_ok=True)
    #print(fbase)
    f=open(fname,"rb")
    frameno=-1
    flowqueue = []
    oldflowp=None
    oldflowv=None
    while True:
        # FLOW PARAMS [u,v,p,entropy]
        # read flow params metadata [rows,cols,visc]
        flowpmeta = np.frombuffer(f.read(4*3),np.float32)
        # parse flow params
        flowp_rows = int(flowpmeta[0])
        flowp_cols = int(flowpmeta[1])
        flowp_visc = flowpmeta[2]
        # read flow
        flowp = np.frombuffer(f.read(4*4*flowp_rows*flowp_cols),np.float32)
        # reshape flow params re size read
        flowp=flowp.reshape(flowp_rows,flowp_cols,4).copy()
        # read video params
        flowvmeta = np.frombuffer(f.read(4*2),np.float32)
        # parse video params
        flowv_rows = int(flowvmeta[0])
        flowv_cols = int(flowvmeta[1])
        # read video
        flowv=np.frombuffer(f.read(flowv_rows*flowv_cols*1),np.ubyte)
        # reshape flow video re size read
        flowv=flowv.reshape(flowv_rows,flowv_cols,1)
        flowv = flowv[::-1,:,:]
        flowv = cv2.resize(flowv, (flowp_cols,flowp_rows))
        # print(f"flowv.shape={flowv.shape}")

        # prepare data
        # anything > zero in the 'video' layer is maxxed to 255
        flowv[flowv>0]=255

        # split out the various vector fields
        flowp0 = flowp[...,0]  # u
        flowp1 = flowp[...,1]  # v
        flowp2 = flowp[...,2]  # pressure
        #flowp3 = flowp[...,3]  # entropy calculated

        flowp0[flowv>0] = 0 # zero out velocity inside obstacles
        flowp1[flowv>0] = 0 #    "  "   "
        flowp2[flowv>0] = 1 # normalize pressure to 1 inside obstacles

        # calculate vorticity
        flowpV = np.arctan(fftconvolve(flowp1,[[-1, 0, 1]],'same')+
                  fftconvolve(flowp0,[[1],[0],[-1]],'same'))

        #print(f"flowpV.shape={flowpV.shape}")
        flowpV[flowv>0] = 0  # no vorticity inside obstacles either

        yield(np.stack((
                flowp0,   # u
                flowp1,   # v
                flowp2,   # p
                flowv,    # obst.
                flowpV    # vorticity (curl)
            ))[np.newaxis,...].copy())



"""
*** CODE TO SPLIT DATA LEFT FOR REFERENCE
                for r in range(KDSIZE,flowp.shape[0]-KDSIZE):
                    for c in range(KDSIZE,flowp.shape[1]-KDSIZE):
                        print('!',end='')
                        yield(
                            oldp[:,:,r-KDSIZE:r+KDSIZE+1,c-KDSIZE:c+KDSIZE+1],
                            newp[:,:,c-KDSIZE:c+KDSIZE+1,c-KDSIZE:c+KDSIZE+1])

         # yield oldflowpV[np.newaxis,np.newaxis,...], flowpV[np.newaxis,np.newaxis,...]

        flowqueue.append((
            flowp.copy(),
            flowv.copy(),
            flowpV.copy(),
            ))
"""


def npydatagenerator(npydata,start=0,test=False):
    # this function generates training (input and target pairs) chosen at random from the npydata
    #if test:
    #    ridx = range(len(npydata)-ITERS)
    #else:
    if True:
        while True:
            ridx = np.random.permutation(len(npydata)-ITERS)
            if np.sum(np.abs(np.diff(ridx))<3) < 3:
                break

    idx = start
    while True:
        yield npydata[ridx[idx]],npydata[(ridx[idx]+int(ITERS+.05)) % len(npydata)]
        idx += 1
        idx %= len(npydata)-int(ITERS+0.05)

def TensorFluidBatch(k,batchsize):
    # this function generates a batch of training data from the data generator k
    # of size batchsize
    fmtra = []
    fmtar = []
    for i in range(batchsize):
        fm_train,fm_target = next(k)
        fm_train=fm_train[:,0:4,...]
        fm_target=fm_target[:,0:3,...]
        fmtra.append(fm_train)
        fmtar.append(fm_target)

    fm_train2 = torch.FloatTensor(np.concatenate(fmtra))
    fm_target2= torch.FloatTensor(np.concatenate(fmtar))

    return fm_train2, fm_target2

class EMLoss(nn.Module):
    # this is the earth mover's distance loss function
    def __init__(self):
        super(EMLoss,self).__init__()

    def forward(self,y,x):
        # this is the forward pass of the loss function
        # y is the predicted output
        # x is the target output
        # cdist is the scipy function for computing the pairwise distance between two sets of points
        return cdist(x[0,0,:,:],y[0,0,:,:])
        d0 = cdist(x.cpu().detach().numpy()[0,0,:,:], y.cpu().detach().numpy()[0,0,:,:])
        d1 = cdist(x.cpu().detach().numpy()[0,1,:,:], y.cpu().detach().numpy()[0,1,:,:])
        d2 = cdist(x.cpu().detach().numpy()[0,2,:,:], y.cpu().detach().numpy()[0,2,:,:])
        assignment0 = linear_sum_assignment(d0)
        assignment1 = linear_sum_assignment(d1)
        assignment2 = linear_sum_assignment(d2)
        emd0 = torch.FloatTensor((d0[assignment0].sum() / np.prod(x.shape),))
        emd1 = torch.FloatTensor((d1[assignment1].sum() / np.prod(x.shape),))
        emd2 = torch.FloatTensor((d2[assignment2].sum() / np.prod(x.shape),))
        return (emd0+emd1+emd2*0.1)*100.0


class SpectralLoss(nn.Module):
    # this is the spectral loss function

    def __init__(self):
        super(SpectralLoss,self).__init__()

    def forward(self,y,x):
        # this is the forward pass of the loss function
        # y is the predicted output
        # x is the target output
        # dct is the scipy function for computing the discrete cosine transform
        yd = dct(y.cpu().detach().numpy(),norm='ortho')
        xd = dct(x.cpu().detach().numpy(),norm='ortho')
        return torch.FloatTensor((np.abs(yd-xd).sum() / np.prod(x.shape),))

    #def forward(self,y,x):
    #    yd = dct.

class L1Loss(nn.Module):
    # this is the L1 loss function

    def __init__(self):
        super(L1Loss,self).__init__()

    def forward(self,y,x):
        # this is the forward pass of the loss function
        # y is the predicted output
        # x is the target output
        return torch.FloatTensor((np.abs(y-x).sum() / np.prod(x.shape),))

class L2Loss(nn.Module):
    # this is the L2 loss function

    def __init__(self):
        super(L2Loss,self).__init__()

    def forward(self,y,x):
        # this is the forward pass of the loss function
        # y is the predicted output
        # x is the target output
        return torch.FloatTensor((np.square(y-x).sum() / np.prod(x.shape),))

def printargs(**args):
    print(args)


class NSLoss(nn.Module):
    # navier-stokes loss function
    def __init__(self):
        super(NSLoss,self).__init__()

    def forward(self,y,x):
        # this is the forward pass of the loss function
        # y is the predicted output
        # x is the target output
        return (torch.exp(((y[:,0:3,...]-x[:,0:3,...]).abs()).sum()/np.prod(y.shape[0:-2])/100000.0)-1.0)*50

def same_padding(in_shape,kernel_size,strides_in):
    # this function computes the padding needed to ensure that the output of a convolutional layer
    # has the same shape as the input
    in_height, in_width = in_shape
    filter_height, filter_width = kernel_size
    strides=(None,strides_in[0],strides_in[1])
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width  = np.ceil(float(in_width) / float(strides[2]))

    # The total padding applied along the height and width is computed as:
    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if (in_width % strides[2] == 0):
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    print(pad_along_height, pad_along_width)
                  
    #Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    print(pad_left, pad_right, pad_top, pad_bottom)
    return(pad_top,pad_left)

class ICNN(nn.Module):
    # this class was arbitrarily named ICNN. The "I" doesn't stand for inverse.
    # This class is intended to be trained with the Navier-Stokes flow equations.
    # The input to the network is a 2D velocity field, a pressure field, and a mask field defining
    # arbitrary boundary geometry. The output is a 2D velocity field and a pressure field.
    # The network is trained to predict the velocity and pressure fields that satisfy the Navier-Stokes
    # equations. The network is trained with the NSLoss function defined above.

    def conv(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=(4,4),leak=0.0,pool_size=1,dropout=0.0):
        #print(f"in_channels:{in_channels},out_channels:{out_channels},kernel_size:{kernel_size},"+
        #        f"stride:{stride},padding:{padding},leak:{leak},pool_size:{pool_size},dropout:{dropout}")
        padding=(kernel_size//2,kernel_size//2)
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(leak),
            nn.MaxPool2d(pool_size),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels)
        )

    def __init__(self,MPD):
        super(ICNN, self).__init__()
        self.MPD = MPD
        self.conv0 = ICNN.conv(
            in_channels=4,
            out_channels=mp(MPD,'LAYER:0:NCHAN',8),
            kernel_size=mp(MPD,'LAYER:0:KSIZE',9),
            leak=mp(MPD,'LAYER:0:LEAK',0.0),
            dropout=mp(MPD,'LAYER:0:DROPOUT',0.0))

        convs_list = nn.ModuleList()
        nlayers = len(mp(MPD,'LAYER'))
        for i in range(1,nlayers):
            convs_list.append(ICNN.conv(
                in_channels=mp(MPD,f'LAYER:{i-1}:NCHAN',8),
                out_channels=mp(MPD,f'LAYER:{i}:NCHAN',8),
                kernel_size=mp(MPD,f'LAYER:{i}:KSIZE',9),
                leak=mp(MPD,f'LAYER:{i}:LEAK',0.0),
                dropout=mp(MPD,f'LAYER:{i}:DROPOUT',0.0)))
        self.convs = nn.Sequential(*convs_list)

        self.convLast = ICNN.conv(
            in_channels=mp(MPD,f'LAYER:{nlayers-1}:NCHAN',8),
            out_channels=3,
            kernel_size=mp(MPD,'LAYER_LAST:KSIZE',9),
            leak=mp(MPD,'LAYER_LAST:LEAK',0.0),
            dropout=mp(MPD,'LAYER_LAST:DROPOUT',0.0))

    def forward(self,x):
        x1 = self.conv0(x)
        x1 = self.convs(x1)
        x1 = self.convLast(x1)

        return x1

def norm(x):
    #x = np.abs(np.fft.ifft2(xp))
    return (x-x.min())/(x.max()-x.min())

def train(num_epochs, cnn, k):
    global losses,lossavgs,lossstds
    losses=[]
    lossavgs=[]
    lossstds=[]
    cnn.cuda()
    bestloss =1000
    beststd = 1000
    try:
        for me in range(num_meta_epochs):
            # Train the model
            total_step = 1 #traindata.shape[0] // BATCHSIZE
            #print(f"total_step = {total_step}")
            optimizer.zero_grad()
            losses=[]
            lossavgs=[]
            lossstds=[]
            state_dict=None
            savedepoch=None
            for epoch in range(num_epochs):
                verb= (epoch % 1 == 0) and VERBOSE
                traindata,targetdata = TensorFluidBatch(k,BATCHSIZE)
                traindata=traindata.cuda()
                targetdata=targetdata.cuda()
                for i in range(0,total_step):
                    if verb: 
                        print (f'Epoch [{epoch + 1}/{num_epochs}], shapein:{traindata.shape}, tr/tg:{traindata[0,0,1,1]:+8f}/{targetdata[0,0,1,1]:+8f}  ',end='',flush=True)
                    else:
                        print(".",end='',flush=True)
                    # gives batch data, normalize x when iterate train_loader
                    #b_x = Variable(images)   # batch x
                    #b_y = Variable(labels)   # batch y
                    #print(f"b_x.shape = {b_x.shape}")
                    #if epoch % BATCHSIZE*4 == 0:
                    optimizer.zero_grad()
                    if verb: print("calc: ",flush=True,end='')
                    output = cnn(traindata)
                    if verb: print(f" shapeout={output.shape} ",end='')
                    if int(ITERS) > 1:
                        for ii in range(int(ITERS)-1):
                            nextput = (cnn(output.to(device)))
                            output = nextput
                    loss = loss_func(output,targetdata[:,0:4,...].to(device))
                    if verb: print("eval, ",flush=True,end='')

                    #loss.requires_grad = True
                    #print(f"traindata.shape={traindata.shape},
                    #output.shape={output.shape}", end='') # clear gradients for
                    #this training step   
                    
                    # backpropagation, compute gradients 
                    if verb: print("bprop, ",flush=True,end='')
                    loss.backward()
                    # apply gradients
                    if verb: print("appgrads, ",flush=True,end='')
                    optimizer.step()
                    if verb: print("done. ",flush=True,end='')

                    losses.append(loss.item())
                    lossavg=np.median(losses[max(0,epoch-20):epoch+1])
                    lossstd=np.std(losses[max(0,epoch-20):epoch+1])
                    lossavgs.append(lossavg)
                    lossstds.append(lossstd)
                    if (i+1) % 1 == 0:
                        print ('{},{},{},{},{:.20f}' 
                               .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()),file=open(f'{datadir}/report.txt','a'))
                        if verb: print (f'log {lossfuncdesc}: {np.log10(loss.item()):.4f}',flush=True,end='')
                    pass
                    if lossstd < beststd and epoch>=20:
                        state_dict = copy.deepcopy(cnn.state_dict())
                        beststd=lossstd
                        savedepoch=epoch+1
                        if verb:print("[saved]",end='')
                    if verb:print("")
                    #j=os.system("nvidia-smi -i 0 -q |grep 'Used GPU' | awk -F: '{ print $2 }' | awk '{ print $1 }'>/tmp/gpumem") 
                    #gpumem = int(open("/tmp/gpumem",'r').read())
                    #print(f"mem used={gpumem}")
                    #if gpumem > 2000:
                    #    sys.exit(0)

                scheduler.step(loss)
                #if loss.item() < 0.0002: break
            if state_dict is not None:
                torch.save(state_dict, f"{datadir}{os.path.sep}cnnstate{savedepoch:06d}.dat")
                if VERBOSE: print(f"saved state of epoch {savedepoch} to {datadir}{os.path.sep}cnnstate{savedepoch:06d}.dat")

    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)

    print("")


def test(pnum_epochs, cnn, k):
    num_epochs = pnum_epochs//5
    global losses,lossavgs,lossstds
    losses=[]
    lossavgs=[]
    lossstds=[]
    cnn.eval()
    cnn.cuda()
    bestloss =1000
    beststd = 1000
    try:
        with torch.no_grad():
            # Test the model
            total_step = 1 #traindata.shape[0] // BATCHSIZE
            #print(f"total_step = {total_step}")
            state_dict=None
            savedepoch=None

            for epoch in range(num_epochs):
                verb= (epoch % 1 == 0) and VERBOSE
                traindata,targetdata = TensorFluidBatch(k,BATCHSIZE)
                traindata=traindata.cuda()
                targetdata=targetdata.cuda()
                for i in range(0,total_step):
                    if verb: 
                        print (f'Epoch [{epoch + 1}/{num_epochs}], shapein:{traindata.shape}, tr/tg:{traindata[0,0,1,1]:+8f}/{targetdata[0,0,1,1]:+8f}  ',end='',flush=True)
                    else:
                        print("t",end='',flush=True)
                    # gives batch data, normalize x when iterate train_loader
                    #b_x = Variable(images)   # batch x
                    #b_y = Variable(labels)   # batch y
                    #print(f"b_x.shape = {b_x.shape}")
                    #if epoch % BATCHSIZE*4 == 0:
                    if verb: print("calc: ",flush=True,end='')
                    output = cnn(traindata)
                    if verb: print(f" shapeout={output.shape} ",end='')
                    if int(ITERS) > 1:
                        for ii in range(int(ITERS)-1):
                            nextput = (cnn(output.to(device)))
                            output = nextput
                    loss = loss_func(output,targetdata[:,0:4,...].to(device))
                    if verb: print("eval, ",flush=True,end='')

                    if verb: print("done. ",flush=True,end='')

                    losses.append(loss.item())
                    lossavg=np.median(losses[max(0,epoch-40):epoch+1])
                    lossstd=np.std(losses[max(0,epoch-40):epoch+1])
                    lossavgs.append(lossavg)
                    lossstds.append(lossstd)
                    if (i+1) % 1 == 0:
                        print ('{},{},{},{},{:.20f}' 
                               .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()),file=open(f'{datadir}/report.txt','a'))
                        if verb: print (f'log {lossfuncdesc}: {np.log10(loss.item()):.4f}',flush=True,end='')
                    pass
                    if lossstd < beststd and epoch>=20:
                        state_dict = copy.deepcopy(cnn.state_dict())
                        beststd=lossstd
                        savedepoch=epoch+1
                        if verb:print("[saved]",end='')
                    if verb:print("")


    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)

    print("")




def fluidtest():
    ktest = npydatagenerator(testdata, test=True)
    global test_data
    # Test the model
    cnn.eval()
    cnn.cuda()
    test_data,test_target = next(ktest)
    while(True):
        with torch.no_grad():
            print(f"testing: in={test_data.shape}")
            test_data = torch.FloatTensor(test_data).cuda()
            output = cnn(test_data).cpu()
            print(f"output={output.shape}")
            # calculate vorticity
            vorticity = np.arctan(fftconvolve(output[0,1,...],[[-1, 0, 1]],'same')+
                      fftconvolve(output[0,0,...],[[1],[0],[-1]],'same'))
            vort=np.zeros(vorticity.shape+(3,))
            vort[...,0] = np.log(np.abs(vorticity)+1) * (vorticity>=0)
            vort[...,1] = 0.0*np.abs(vorticity) * (vorticity>=0)
            vort[...,2] = np.log(np.abs(vorticity)+1) * (vorticity<0)
            vort=norm(vort)
            cv2.imshow('test',vort)
            ky = cv2.waitKey(5000)
            if ky in [3,27,81,113]: break
            if ky in [ord('n'),ord("N")]:
                test_data,test_target=next(ktest)
                
            else:
                next_data,test_target=next(ktest)
                next_data[:,0:2,...] = output.cpu()[:,0:2,...]*1.0
                next_data[:,3:4,...] = 0
                test_data = torch.FloatTensor(next_data.copy())
        pass
    cv2.destroyAllWindows()


def walktemplate(templ,MP,prefix=''):
    # this function walks the template and calls mp() for each parameter
    if (isinstance(templ,list) and len(templ)==3 and
        (isinstance(templ[0],int) or isinstance(templ[0],float)) and
        (isinstance(templ[1],int) or isinstance(templ[1],float)) and
        (isinstance(templ[2],int) or isinstance(templ[2],float))):
        pfx=prefix[:-1]
        pnstr = pfx
        if DEBUG: print(f"{pfx}:{templ} => ",end='')
        if DEBUG: print(mp(MP,pfx),end='')
        pv = mp(MP,pfx) 

        pmin=templ[0]
        pmax=templ[1]
        pint=templ[2]
        numint = np.floor((pmax-pmin)/pint) + 1
        if np.random.randint(0,mutation_period)==1:
            pv=pv+(np.random.randint(-3,4))*pint
            if DEBUG: print(" ~ ",end='')
        else:
            if DEBUG: print(" = ",end='')
        if pv<pmin: pv=pmin
        if pv >= pmax: pv=pmax-pint
        pv=np.around(pv,6)
        if DEBUG: print(pv)
        mp(MP,pfx,force=pv)
        if DEBUG: print(mp(MP,pfx))

    elif isinstance(templ,list):
        for i in range(len(templ)):
            walktemplate(templ[i],MP,prefix=prefix+f"{i}:")
    elif isinstance(templ,dict):
        for key in templ:
            walktemplate(templ[key],MP,prefix=prefix+f"{key}:")
    else:
        pass
        #print(f"{prefix}:{templ}")


def _agent_serialize(obj,agent, prefix=''):
    
    if isinstance(agent,list):
        for i in range(len(agent)):
            _agent_serialize(obj,agent[i],prefix=prefix+f"{i}:")
    elif isinstance(agent,dict):
        for key in agent:
            _agent_serialize(obj,agent[key],prefix=prefix+f"{key}:")
    elif isinstance(agent,int) or isinstance(agent,float) or isinstance(agent,str):
        obj[prefix[:-1]] = agent
    else:
        print(f"WARNING: agent element {prefix} is of illegal type {type(agent)}")
        pass

def agent_serialize(agent):
    obj = {}
    _agent_serialize(obj, agent,prefix='')
    if DEBUG: print(obj)
    return obj

def agent_deserialize(ser_agent):
    newagent = copy.deepcopy(THISGEN['TEMPLATE'])
    for dolt in ser_agent.keys():
        if DEBUG: print(f"mp(newagent,{dolt},force={ser_agent[dolt]})")
        mp(newagent,dolt,force=ser_agent[dolt])
    return newagent

def diploid(agent1, agent2):
    cagent1 = copy.deepcopy(agent1)
    cagent2 = copy.deepcopy(agent2)
    if 'REPORT' in cagent1: del cagent1['REPORT']
    if 'REPORT' in cagent2: del cagent2['REPORT']
    seragent1 = agent_serialize(cagent1)
    seragent2 = agent_serialize(cagent2)
    seragent3 = copy.deepcopy(seragent1)

    for dolt in seragent3.keys():
        fromagent1 = seragent1[dolt]
        fromagent2 = seragent2[dolt]
        if np.random.randint(0,2)==0:
            seragent3[dolt] = fromagent1
        else:
            seragent3[dolt] = fromagent2
    if DEBUG: print("=== diploid result ===")
    if DEBUG: print(seragent3)
    if DEBUG: print("=== diploid deserialized ===")
    k = agent_deserialize(seragent3)
    if DEBUG: print(k)
    return k


# *** LOAD THE DATASET ***
def load_data(nrecs=args.loadrecs):
    # this function loads the dataset from the npy files
    npyfiledata = []
    print("LOADING DATA")
    ftras=sorted(glob(f'{NPYDIR}{os.path.sep}lbinfo*.npy'))
    npylist = []
    recsloaded=0
    for ftra in ftras:
        if os.path.exists(ftra):
            npyfiledata = np.load(ftra)
            print(".",end='',flush=True)
            print(len(npyfiledata))
            if len(npyfiledata)+recsloaded > nrecs:
                npylist += [ npyfiledata[:nrecs-recsloaded] ]
                recsloaded = nrecs
                break
            npylist += [npyfiledata]
            recsloaded += len(npyfiledata)
   
    alldata = np.concatenate(npylist,axis=0)
    print(f"DATA LOADED: {len(alldata)}")
    return alldata

def convert_lbinfo_npy():
    # this function converts the lbinfo file into a series of .npy files
    print("makin' .npy's")
    k0 = lbinfodatagenerator()
    print("generating data")
    idx=0 # count of records
    fidx=0 # count of .npy files created
    stacked = []
    exiting = False
    while True:
        # *** load the next record from lbinfo file ***
        try:
            ftrain = next(k0)
            stacked.append(ftrain)
            if idx % 100 == 0:
                print(idx,flush=True)
            idx += 1
        except KeyboardInterrupt as e:
            print("=== KEYBOARD STOP ===")
            exiting = True
        except Exception as e:
            print("Exception in user code:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            exiting = True

        if len(stacked)>= 1000 or exiting:
            os.makedirs(NPYDIR,exist_ok=True)
            np.save(f'{NPYDIR}{os.path.sep}lbinfo{fidx:04d}.npy',np.stack(stacked))
            print(  f"{NPYDIR}{os.path.sep}lbinfo{fidx:04d}.npy")
            del stacked
            gc.collect()
            stacked = []
            fidx += 1
            if exiting:
                break

    sys.exit(0)


# === END OF DEFINES ===

TRAINING=True
TESTING= ('-t' in sys.argv or '-tr' in sys.argv or '-rt' in sys.argv)
RESUMING=False # '-r' in sys.argv or '-tr' in sys.argv or '-rt' in sys.argv
# *** ESTABLISH COMPUTING DEVICE ***

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"COMPUTING DEVICE: {device}")

# *** OPTION TO GENERATE npy FILE FROM DATA ***
if args.npy: 
    convert_lbinfo_npy()
    # note: above call never returns; -npy is like a standalone utility

#*** load npy file data ***
npydata = load_data()

testdata = npydata[:len(npydata)*1//4]
npydata = npydata[len(npydata)*1//4:]


# default behavior is to continue where we left off. 
# however there needs to be overrides to that behavior, which
# need to be taken care of first.

# *** OPTION TO ARCHIVE PREVIOUS EVOLUTION AND START WITH NEW EVODIR ***
# we don't need to "archive" as much as "close the books on" the current evo model if any.

# current evo model lives in:
#    evo/current_evo/evo_{timestamp}/

# to make the current evo no longer current, it simply needs to be moved
# to the evo/prev_evos/ directory.

# the new evo model will be created in the evo/current_evo/ directory
# and will be named evo_{timestamp}/

if args.new_evo:
    # must close out the old evolution if it exists.
    #first check to see if there's an evo there
    evo_paths = glob("evo/current_evo/evo_*")
    if len(evo_paths) > 0:
        evo_path = evo_paths[0]
        evo_id = os.path.basename(evo_path)
        os.makedirs("evo/prev_evos",exist_ok=True)
        os.rename(evo_path,f"evo/prev_evos/{evo_id}")

# determine if a current evo is in progress
evo_starting_template_path = "evo_starting_template.json"
evo_primary_template_path = "evo/evo_primary_template.json"
if not os.path.exists(evo_primary_template_path):
    if not os.path.exists(evo_starting_template_path):
        print(f"no {evo_primary_template_path} or even {evo_starting_template_path}.")
        print("GIVING UP.")
        sys.exit(0)
    else:
        os.makedirs("evo",exist_ok=True)
        shutil.copy(evo_starting_template_path, evo_primary_template_path)

evo_paths = glob("evo/current_evo/evo_*")
if len(evo_paths) > 0:
    # yes there is. load up the path & ID info
    evo_path = evo_paths[0]
    evo_id = os.path.basename(evo_path)
    evo_template_path = evo_path+"/evo_template.json"
    if not os.path.exists(evo_template_path):
        shutil.copy("evo/evo_primary_template.json",evo_template_path)
else:
    # no current evo, must create one
    print(f"no current evo, must create one.")
    # to be sure
    os.makedirs("evo/current_evo",exist_ok=True)
    evo_id = "evo_"+get_timestamp()
    evo_path=f"evo/current_evo/{evo_id}"
    os.makedirs(evo_path,exist_ok=True)
    evo_template_path = evo_path+"/evo_template.json"
    if not os.path.exists(evo_template_path):
        shutil.copy("evo/evo_primary_template.json",evo_template_path)

# current evo is established, now:
while True: # loop through generations
    # is there a generation in progress?
    gen_paths = glob(evo_path+"/current_gen/gen_*")
    if len(gen_paths) > 0:
        gen_path = gen_paths[0]
        gen_id = os.path.basename(gen_path)
        gen_working_path = gen_path+"/gen_population_working.json"
    else:
        # make a generation
        gen_id = "gen_"+get_timestamp()
        gen_path = evo_path+"/current_gen/"+gen_id
        gen_working_path = gen_path+"/gen_population_working.json"
        os.makedirs(gen_path,exist_ok=True)
        shutil.copy(evo_template_path,gen_working_path)
        # this is saved as a record of what the template was at the time this generation was run
        shutil.copy(evo_template_path,gen_path+"/gen_evo_template.json")

    THISGEN = json.load(open(gen_working_path,"r"))
    pop_size = THISGEN['POP_SIZE']
    num_survive = THISGEN['NUM_SURVIVE']
    num_breed = THISGEN['NUM_BREED']
    mutation_period = THISGEN['MUTATION_PERIOD']


    ## loop through agents ##
    while True:
        # this must act like a state machine so that we can be interrupted
        # and pick up where we left off easily.
        if 'AGENTS_RUN' in THISGEN:
            if len(THISGEN['AGENTS_RUN']) >= pop_size:
                # we are DONE.
                break
        if 'AGENTS_IN' in THISGEN and len(THISGEN['AGENTS_IN']) > 0:
            agents_in = THISGEN['AGENTS_IN']
            agent_id = list(agents_in.keys())[0]
            MP=agents_in[agent_id]
        else:
            # create a new agent
            agent_id = get_timestamp()
            if 'TEMPLATE' in THISGEN:
                MP = copy.deepcopy(THISGEN['TEMPLATE'])
            else:
                MP={}
        MP['AGENT_ID'] = agent_id
        datadir=f"{gen_path}/{agent_id}"
        if DEBUG: print("---- raw template ----")
        if DEBUG: print(MP)
        if DEBUG: print("---walking---")
        walktemplate(THISGEN['TEMPLATE'],MP)
        if DEBUG: print("--- walked ---")
        if DEBUG: print(MP)
        BATCHSIZE=mp(MP,'BATCHSIZE',16)
        num_epochs = THISGEN['NUM_EPOCHS']
        num_meta_epochs = mp(MP,'NUM_META_EPOCS',1)
        ITERS=mp(MP,'ITERS',1)
        SCRAMBLE_SIZE=mp(MP,'SCRAMBLE_SIZE',420)
        os.environ['CUDA_LAUNCH_BLOCKING']='1'

        np.random.seed()

        k=npydatagenerator(npydata, np.random.randint(0,num_epochs))


        datadir = f"{gen_path}/{agent_id}"
        os.makedirs(datadir, exist_ok=True)
        if TRAINING and DEBUG:
            print(f"TRAINING: datadir={datadir}")

        #if (not TRAINING) or RESUMING:
        #    sep=os.path.sep
        #    resdatadir= sep.join(sorted(glob(f"{DATADIR}{sep}*{sep}cnnstate.dat"))[-resumeskip].split(sep)[:-1])
        #    print(f"Loading pre-trained weights from: datadir={resdatadir}")

        cnn = ICNN(MP)
        #print(cnn)

        loss_func = nn.MSELoss()
        lossfuncdesc="MSELoss"
        if False and int(mp(MP,'ITERS')*10+0.5)//2 *2 == int(mp(MP,'ITERS')*10+0.5):
            loss_func=nn.L1Loss()
            lossfuncdesc="L1Loss"
        if DEBUG: print(f"LOSS: {lossfuncdesc}")

        optlr = 10**(mp(MP,'INIT_LR',-2))
        optimizer = optim.Adam(cnn.parameters(), lr = optlr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.5,patience=5)




        #if (not TRAINING) or RESUMING:
        #    try:
        #        cnn.load_state_dict(torch.load(f"{resdatadir}{os.path.sep}cnnstate.dat"))
        #    except RuntimeError as e:
        #        pass
        #    cnn.cuda()
        #    print(f"loaded {resdatadir}{os.path.sep}cnnstate.dat")

        cnn.cuda()
        cnn.train().cuda()
        #summary(cnn.cuda(), (1,1,175,256))
        open(f"{datadir}{os.path.sep}model_params.json","w").write(json.dumps(MP,indent=4))
        #open(f"model_params.json","w").write(json.dumps(MP,indent=4))
        if TRAINING:
            if DEBUG: print("training.")
            train(num_epochs, cnn, k)
            torch.save(cnn.state_dict(), f"{datadir}{os.path.sep}cnnfinalstate.dat")
            if DEBUG: print(f"saved state to {datadir}{os.path.sep}cnnfinalstate.dat")
            #print(f"copy ({sys.argv[0]},{datadir}{os.path.sep}torchfluid{get_timestamp()}.py)")
            shutil.copy(sys.argv[0],   f"{datadir}{os.path.sep}torchfluid{get_timestamp()}.py")

        # training completed!
        if 'AGENTS_RUN' not in THISGEN:
            THISGEN['AGENTS_RUN']={}
        if 'AGENTS_IN' not in THISGEN:
            THISGEN['AGENTS_IN']={}
        THISGEN['AGENTS_RUN'][agent_id] = MP.copy()
        THISGEN['AGENTS_RUN'][agent_id]['REPORT'] = {
                'losses': losses,
                'lossavgs': lossavgs,
                'lossstds': lossstds
                }
        testk = npydatagenerator(testdata,test=True)
        test(num_epochs, cnn, testk)
        THISGEN['AGENTS_RUN'][agent_id]['REPORT'] = {
                'losses': losses,
                'lossavgs': lossavgs,
                'lossstds': lossstds
                }



        if agent_id in THISGEN['AGENTS_IN']:
            del THISGEN['AGENTS_IN'][agent_id]

        json.dump(THISGEN,open(f'{gen_path}/_IN_gen_population_working.json','w'),indent=4)
        os.replace(f'{gen_path}/_IN_gen_population_working.json',f'{gen_path}/gen_population_working.json')

    if len(THISGEN['AGENTS_RUN']) >= pop_size:
        if os.path.exists("stop.flg"):
            os.remove("stop.flg")
            sys.exit(0)
        #** DONE WITH THIS GENERATION **
        # set up for next generation
        scores = []
        for agent in THISGEN['AGENTS_RUN']:
            scorelist = np.array(THISGEN['AGENTS_RUN'][agent]['REPORT']['lossavgs'])
            # calculate projected score in 50 epochs by going back 50 epochs and projecting
            realendscore = scorelist[-1]
            scores.append({'agent':agent,'score':realendscore })
        scores=pd.DataFrame(scores)
        scores=scores.sort_values('score')
        if DEBUG: print(scores)
        scores.to_csv(f"{gen_path}/scores.csv")
        if TESTING:
            MP=copy.deepcopy(THISGEN['AGENTS_RUN'][scores.loc[0,'agent']])
            cnn = ICNN(MP)
            try:
                cnnstatefile = glob(f"{DATADIR}{os.path.sep}{scores.loc[0,'agent']}{os.path.sep}cnnstate*.dat")[0]
                cnn.load_state_dict(torch.load(cnnstatefile))
            except RuntimeError as e:
                print(f"Exception in user code: {e}")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                sys.exit(0)
                pass
            cnn.cuda()
            print(f"loaded {cnnstatefile}")
            print(f"last score: {scores.loc[0,'score']}")
            input("About to run test - Press ENTER:")
            test()
            print("=== done with test, press enter for next generation ===")
            input("Press ENTER: ")


        scores.drop(1)
        breeders  = scores.iloc[:THISGEN['NUM_BREED'],:]
        survivors = scores.iloc[THISGEN['NUM_BREED']:THISGEN['NUM_SURVIVE'],:]
        if DEBUG: print('--breed--')
        if DEBUG: print(breeders)
        if DEBUG: print('--survive--')
        if DEBUG: print(survivors)
        if not os.path.exists(f'{gen_working_path}'):
            print(f"FATAL: {gen_working_path} missing")
            sys.exit(0)

        NEXTGEN=json.load(open(f'{evo_template_path}','r'))
        NEXTGEN['NUM_EPOCHS'] = NEXTGEN['NUM_EPOCHS']+NEXTGEN['EPOCH_INC']
        json.dump(NEXTGEN,open(f'{evo_template_path}','w'),indent=2)
        json.dump
        if 'AGENTS_IN' not in NEXTGEN:
            NEXTGEN['AGENTS_IN'] = {}
        breedlist = list(breeders['agent'])
        while len(breedlist) > 0:
        #for i in range(0,len(breedlist)//2):
            newspawn = get_timestamp()
            p1 = 0
            p2 = int(abs(np.random.normal()*len(breedlist)/4))+1
            if p2 >= len(breedlist):
                p2 = len(breedlist)-1
            
            print(f"spawning {breedlist[p1]} x {breedlist[p2]}")
            NEXTGEN['AGENTS_IN'][childname(breedlist[p1])] =   diploid(THISGEN['AGENTS_RUN'][breedlist[p1]],
                                                                      THISGEN['AGENTS_RUN'][breedlist[p2]])
            if DEBUG: print(childname(breedlist[p1]))
            if DEBUG: print(NEXTGEN['AGENTS_IN'][childname(breedlist[p1])]) #['AGENT_ID'] = childname(breedlist[i])
            NEXTGEN['AGENTS_IN'][childname(breedlist[p1])]['AGENT_ID'] = childname(breedlist[p1])
            NEXTGEN['AGENTS_IN'][childname(breedlist[p2])] = diploid(THISGEN['AGENTS_RUN'][breedlist[p1]],
                                                                      THISGEN['AGENTS_RUN'][breedlist[p2]])
            NEXTGEN['AGENTS_IN'][childname(breedlist[p2])]['AGENT_ID'] = childname(breedlist[p2])
            #del NEXTGEN['AGENTS_IN'][breedlist[i]]['REPORT']
            #del NEXTGEN['AGENTS_IN'][breedlist[i+1]]['REPORT']
            NEXTGEN['AGENTS_IN'][breedlist[p1]] = copy.deepcopy(THISGEN['AGENTS_RUN'][breedlist[p1]])
            del NEXTGEN['AGENTS_IN'][breedlist[p1]]['REPORT']
            del breedlist[p2]
            del breedlist[p1]
        """ OLD MYTOSIS METHOD 
        for breeder in list(breeders['agent']):
            NEXTGEN['AGENTS_IN'][breeder] = copy.deepcopy(THISGEN['AGENTS_RUN'][breeder])
            NEXTGEN['AGENTS_IN'][childname(breeder)] = copy.deepcopy(THISGEN['AGENTS_RUN'][breeder])
            NEXTGEN['AGENTS_IN'][childname(breeder)]['AGENT_ID']=childname(breeder)
            del NEXTGEN['AGENTS_IN'][breeder]['REPORT']
            del NEXTGEN['AGENTS_IN'][childname(breeder)]['REPORT']
        """
        for survivor in list(survivors['agent']):
            NEXTGEN['AGENTS_IN'][survivor] = copy.deepcopy(THISGEN['AGENTS_RUN'][survivor])
            del NEXTGEN['AGENTS_IN'][survivor]['REPORT']

        if DEBUG: print(json.dumps(NEXTGEN, indent=2))

        # cause mutations
        # must walk the template

        template=NEXTGEN['TEMPLATE']
        agentids=list(NEXTGEN['AGENTS_IN'].keys())
        #agentids=agentids[NEXTGEN['NUM_BREED']:]
        for ag in agentids[3:]:
            walktemplate(template, NEXTGEN['AGENTS_IN'][ag])
        json.dump(NEXTGEN,open(f'{evo_path}/next_gen.json','w'),indent=2)

        # move old generation out of current_gen and into prev_gens
        gen_dest = f"{evo_path}/prev_gens/{gen_id}"
        os.makedirs(f"{evo_path}/prev_gens",exist_ok=True)
        os.rename(gen_path,gen_dest)
        # make new generation

        gen_id = f"gen_{get_timestamp()}"
        gen_path = f"{evo_path}/current_gen/{gen_id}"
        gen_working_path = gen_path+"/gen_population_working.json"
        os.makedirs(gen_path)
        os.rename(f"{evo_path}/next_gen.json",f"{gen_working_path}")
        shutil.copy(evo_template_path,f"{gen_path}/gen_evo_template.json")
        THISGEN = json.load(open(gen_working_path,"r"))
        pop_size = THISGEN['POP_SIZE']
        num_survive = THISGEN['NUM_SURVIVE']
        num_breed = THISGEN['NUM_BREED']
        mutation_period = THISGEN['MUTATION_PERIOD']
    else:
        break

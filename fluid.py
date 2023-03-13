#!/usr/bin/env python3
import sys,os,json
import numpy as np
import matplotlib.pyplot as plt
import traceback
import cv2
import torch
import torch.nn as nn
import time
from glob2 import glob
timestamp = time.strftime("%Y%m%d_%H%M%S")
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve
import pandas as pd
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"COMPUTING DEVICE: {device}")


if 'MODEL_PARAMS' in os.environ and os.path.exists(os.environ['MODEL_PARAMS']):
    MP = json.load(open(os.environ['MODEL_PARAMS'],"r"))
elif os.path.exists('template_model_params.json'):
    MP = json.load(open('template_model_params.json','r'))
elif os.path.exists('model_params.json'):
    MP = json.load(open('model_params.json','r'))
else:
    MP = {}
print(MP)

def mp(pn,dflt=None):
    pns = pn.split(":")
    for i in range(len(pns)):
        if pns[i].isnumeric():
            pass
        else:
            pns[i]='"'+pns[i]+'"'

    pnstr = "MP["+"][".join(pns)+"]"
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
            exec(pnstr + f" = {paramval}")
        #print(f"={paramval}")
        return paramval
    except Exception as e:
        print(f"ERROR {e}: "+pnstr)
        return dflt

TRAINING=not ('-t' in sys.argv or '-tr' in sys.argv)
RESUMING='-r' in sys.argv or '-tr' in sys.argv
CLEARING='-newdata' in sys.argv
resumeskip=1
if sys.argv[-1].isnumeric():
    resumeskip=int(sys.argv[-1])
BATCHSIZE=mp('BATCHSIZE',16)
num_epochs = mp('NUM_EPOCHS',4000)
num_meta_epochs = mp('NUM_META_EPOCS',1)
ITERS=mp('ITERS',1)
SCRAMBLE_SIZE=mp('SCRAMBLE_SIZE',420)
os.environ['CUDA_LAUNCH_BLOCKING']='1'
#import torch_dct as dct

#from torchvision import datasets
#from torchvision.transforms import ToTensor
#from torchinfo import summary

if CLEARING:
    if os.path.exists("datadir"):
        os.makedirs(f"archived_data",exist_ok=True)
        os.rename("datadir",f"archived_data/datadir{timestamp}")
        os.makedirs("datadir",exist_ok=True)


def datagenerator():
    fname = sorted(glob("../lbinfo-*.csv"))[-1]
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
        flowv[flowv>0]=255

        flowp0 = flowp[...,0]
        flowp1 = flowp[...,1]
        flowp2 = flowp[...,2]
        flowp3 = flowp[...,3]

        flowp0[flowv>0] = 0
        flowp1[flowv>0] = 0
        flowp2[flowv>0] = 1

        # calculate vorticity
        flowpV = np.arctan(fftconvolve(flowp1,[[-1, 0, 1]],'same')+
                  fftconvolve(flowp0,[[1],[0],[-1]],'same'))

        #print(f"flowpV.shape={flowpV.shape}")
        flowpV[flowv>0] = 0

        if len(flowqueue) == int(ITERS):

            """
            yield (
                np.stack((
                oldflowp[...,0],
                oldflowp[...,1],
                oldflowp[...,2],
                oldflowpV,
                oldflowv))[np.newaxis,...].copy(),
                np.stack((
                flowp[...,0],
                flowp[...,1],
                flowp[...,2],
                flowpV
                ))[np.newaxis,...].copy())
            """
            (oldflowp, oldflowv,oldflowpV) = flowqueue.pop()
            yield oldflowpV[np.newaxis,np.newaxis,...], flowpV[np.newaxis,np.newaxis,...]
        flowqueue.append((
            flowp.copy(),
            flowv.copy(),
            flowpV.copy(),
            ))

if '-npy' in sys.argv:  # eventually make this a feature
    k0 = datagenerator()

    idx=0
    while True:
        try:
            ftrain,ftarg = next(k0)
            np.save(f'npy{os.path.sep}tra{idx:07d}.npy',ftrain)
            #np.save(f'npy{os.path.sep}tar{idx:07d}.npy',ftarg)
            print(".",end='',flush=True)
            if idx % 80 == 0:
                print(idx,flush=True)
            idx += 1
        except:
            break


npydata = []
idx=0
print("LOADING DATA")
while True:
    ftra=f'npy{os.path.sep}tra{idx:07d}.npy'
    if os.path.exists(ftra):
        npydata.append(np.load(ftra))
        print(".",end='',flush=True)
        idx += 1
        if idx % 100 == 0:
            print(idx,flush=True)
    else:
        break
print(f"DATA LOADED: {len(npydata)}")

np.random.seed()

def npydatagenerator(start=0):
    while True:
        ridx = np.random.permutation(len(npydata))
        if np.sum(np.abs(np.diff(ridx))<3) < 3:
            break

    idx = start
    while True:
        yield npydata[ridx[idx]],npydata[(ridx[idx]+int(ITERS+.05)) % len(npydata)]
        idx += 1
        idx %= len(npydata)


k=npydatagenerator(np.random.randint(0,num_epochs))

def TensorFluidBatch(k,batchsize):
    fmtra = []
    fmtar = []
    for i in range(batchsize):
        fm_train,fm_target = next(k)
        fmtra.append(fm_train)
        fmtar.append(fm_target)

    fm_train2 = torch.FloatTensor(np.concatenate(fmtra))
    fm_target2= torch.FloatTensor(np.concatenate(fmtar))

    return fm_train2, fm_target2

datadir = f"datadir{os.path.sep}{timestamp}"
os.makedirs(datadir, exist_ok=True)
if TRAINING:
    print(f"TRAINING: datadir={datadir}")

if (not TRAINING) or RESUMING:
    sep=os.path.sep
    resdatadir= sep.join(sorted(glob(f"datadir{sep}*{sep}cnnstate.dat"))[-resumeskip].split(sep)[:-1])
    print(f"Loading pre-trained weights from: datadir={resdatadir}")

class EMLoss(nn.Module):
    def __init__(self):
        super(EMLoss,self).__init__()

    def forward(self,y,x):
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


def printargs(**args):
    print(args)

class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss,self).__init__()

    def forward(self,y,x):
        return (torch.exp(((y[:,0:3,...]-x[:,0:3,...]).abs()).sum()/np.prod(y.shape[0:-2])/100000.0)-1.0)*50















def same_padding(in_shape,kernel_size,strides_in):

    in_height, in_width = in_shape
    filter_height, filter_width = kernel_size
    strides=(None,strides_in[0],strides_in[1])
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width  = np.ceil(float(in_width) / float(strides[2]))

#The total padding applied along the height and width is computed as:

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
    def conv(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=(4,4),leak=0.0,pool_size=1,dropout=0.0):
        print(f"in_channels:{in_channels},out_channels:{out_channels},kernel_size:{kernel_size},"+
                f"stride:{stride},padding:{padding},leak:{leak},pool_size:{pool_size},dropout:{dropout}")
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




    def __init__(self):
        super(ICNN, self).__init__()

        self.conv0 = ICNN.conv(
            in_channels=1,
            out_channels=mp('LAYER:0:NCHAN',8),
            kernel_size=mp('LAYER:0:KSIZE',9),
            leak=mp('LAYER:0:LEAK',0.0),
            dropout=mp('LAYER:0:DROPOUT',0.0))

        convs_list = nn.ModuleList()
        nlayers = len(mp('LAYER'))
        for i in range(1,nlayers):
            convs_list.append(ICNN.conv(
                in_channels=mp(f'LAYER:{i-1}:NCHAN',8),
                out_channels=mp(f'LAYER:{i}:NCHAN',8),
                kernel_size=mp(f'LAYER:{i}:KSIZE',9),
                leak=mp(f'LAYER:{i}:LEAK',0.0),
                dropout=mp(f'LAYER:{i}:DROPOUT',0.0)))
        self.convs = nn.Sequential(*convs_list)

        self.convLast = ICNN.conv(
            in_channels=mp(f'LAYER:{nlayers-1}:NCHAN',8),
            out_channels=1,
            kernel_size=mp('LAYER_LAST:KSIZE',9),
            leak=mp('LAYER_LAST:LEAK',0.0),
            dropout=mp('LAYER_LAST:DROPOUT',0.0))



    def forward(self,x):
        x1 = self.conv0(x)
        x1 = self.convs(x1)
        x1 = self.convLast(x1)

        return x1



cnn = ICNN()
#print(cnn)

loss_func = nn.MSELoss()
lossfuncdesc="MSELoss"
if int(mp('ITERS')*10+0.5)//2 *2 == int(mp('ITERS')*10+0.5):
    loss_func=nn.L1Loss()
    lossfuncdesc="L1Loss"
print(f"LOSS: {lossfuncdesc}")

from torch import optim
optlr = mp('INIT_LR',0.001)
optimizer = optim.Adam(cnn.parameters(), lr = optlr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.5,patience=5)

from torch.autograd import Variable

def norm(x):
    #x = np.abs(np.fft.ifft2(xp))
    return (x-x.min())/(x.max()-x.min())




def train(num_epochs, cnn, k):
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
            for epoch in range(num_epochs):
                traindata,targetdata = TensorFluidBatch(k,BATCHSIZE)
                traindata=traindata.cuda()
                targetdata=targetdata.cuda()
                #plt.imsave(f"{datadir}/{epoch:07d}tx0.png",norm(traindata[0,0,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}tx1.png",norm(traindata[0,1,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}tx2.png",norm(traindata[0,2,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}tx3.png",norm(traindata[0,3,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}ty0.png",norm(targetdata[0,0,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}ty1.png",norm(targetdata[0,1,...].cpu().detach().numpy()))
                #plt.imsave(f"{datadir}/{epoch:07d}ty2.png",norm(targetdata[0,2,...].cpu().detach().numpy()))
                for i in range(0,total_step):
                    print (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], tr/tg:{traindata[0,0,20,20]:+8f}/{targetdata[0,0,20,20]:+8f} ',end='',flush=True)
                    # gives batch data, normalize x when iterate train_loader
                    #b_x = Variable(images)   # batch x
                    #b_y = Variable(labels)   # batch y
                    #print(f"b_x.shape = {b_x.shape}")
                    #if epoch % BATCHSIZE*4 == 0:
                    optimizer.zero_grad()
                    print("calc: ",flush=True,end='')
                    output = cnn(traindata)
                    if int(ITERS) > 1:
                        for ii in range(int(ITERS)-1):
                            nextput = (cnn(output.to(device)))
                            output = nextput
                    loss = loss_func(output,targetdata[:,0:4,...].to(device))
                    print("eval, ",flush=True,end='')

                    #loss.requires_grad = True
                    #print(f"traindata.shape={traindata.shape},
                    #output.shape={output.shape}", end='') # clear gradients for
                    #this training step   
                    
                    # backpropagation, compute gradients 
                    print("bprop, ",flush=True,end='')
                    loss.backward()
                    # apply gradients
                    print("appgrads, ",flush=True,end='')
                    optimizer.step()
                    print("done. ",flush=True,end='')

                    losses.append(loss.item())
                    lossavg=np.mean(losses[max(0,epoch-20):epoch+1])
                    lossstd=np.std(losses[max(0,epoch-20):epoch+1])
                    lossavgs.append(lossavg)
                    lossstds.append(lossstd)
                    if (i+1) % 1 == 0:
                        print ('{},{},{},{},{:.20f}' 
                               .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()),file=open(f'{datadir}/report.txt','a'))
                        print (f'log {lossfuncdesc}: {np.log10(loss.item()):.4f}',flush=True)
                    pass
                    if lossstd < beststd and epoch>=20:
                        torch.save(cnn.state_dict(), f"{datadir}{os.path.sep}cnnstate.dat")
                        print(f"saved state to {datadir}{os.path.sep}cnnstate.dat",flush=True)
                        beststd=lossstd
                    #j=os.system("nvidia-smi -i 0 -q |grep 'Used GPU' | awk -F: '{ print $2 }' | awk '{ print $1 }'>/tmp/gpumem") 
                    #gpumem = int(open("/tmp/gpumem",'r').read())
                    #print(f"mem used={gpumem}")
                    #if gpumem > 2000:
                    #    sys.exit(0)

                scheduler.step(loss)
                #if loss.item() < 0.0002: break

    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)



if (not TRAINING) or RESUMING:
    try:
        cnn.load_state_dict(torch.load(f"{resdatadir}{os.path.sep}cnnstate.dat"))
    except RuntimeError as e:
        pass
    cnn.cuda()
    print(f"loaded {resdatadir}{os.path.sep}cnnstate.dat")

cnn.cuda()
cnn.train().cuda()
#summary(cnn.cuda(), (1,1,175,256))
open(f"{datadir}{os.path.sep}model_params.json","w").write(json.dumps(MP,indent=4))
open(f"model_params.json","w").write(json.dumps(MP,indent=4))
if TRAINING:
    print("training.")
    train(num_epochs, cnn, k)
    torch.save(cnn.state_dict(), f"{datadir}{os.path.sep}cnnfinalstate.dat")
    print(f"saved state to {datadir}{os.path.sep}cnnfinalstate.dat")


ktest = npydatagenerator()


def test():
    global test_data
    # Test the model
    cnn.eval()
    cnn.cuda()
    test_data,test_target = TensorFluidBatch(ktest,1)
    while(True):
        with torch.no_grad():
            test_data[50:70,50:70] = 0
            test_data = test_data.cuda()
            output = cnn(test_data)
            vorticity = test_data[0,0,...]
            vorticity = vorticity.cpu().detach().numpy()
            vort=np.zeros(vorticity.shape+(3,))
            vort[...,0] = np.abs(vorticity) * (vorticity>=0)
            vort[...,1] = np.abs(vorticity) * (vorticity>=0)
            vort[...,2] = np.abs(vorticity) * (vorticity<0)
            vort=norm(vort)

            cv2.imshow('test2',vort)
            ky = cv2.waitKey(5000)
            if ky in [3,27,81,113]: break
            if ky in [ord('n'),ord("N")]:
                test_data,test_target=TensorFluidBatch(ktest,1)
            else:
                test_data = output
        pass
    cv2.destroyAllWindows()

if not TRAINING:
    test()


"""
Poor man's earth mover distance
(actually it's not that bad!! after reading the article I wasn't that far off,
in fact this might be better! LOL!


consider binary distributions:

A: 0 1 0 0 0 0 0 0
B: 0 0 1 0 0 0 0 0
C:_0 0 0 0 0 0 1 0  

d(A,C) denotes the distance value between A and C.

Intuitively, d(A,B) < d(B,C) < d(A,C)

Appplying simple sum of squared differences:

d(A,B) = sum( 0 1 1 0 0 0 0 0 ) = 2
d(B,C) = sum( 0 0 1 0 0 0 1 0 ) = 2
d(A,C) = sum( 0 1 0 0 0 0 1 0 ) = 2

The primary goal is to always have a gradient present such that for a set of three distributions,
the two that are "closer" or "more similar" will have a smaller distance between them.  A simple
um of squared differemces clearly fails this test.

let's consider a very simple "blur" operation which applies a simple convolution kernel 
[ 1 2 1 ] to the three distributions above, with zero-padding applied to preserve the size of
each distribution:

A': 1 2 1 0 0 0 0 0
B': 0 1 2 1 0 0 0 0
C': 0 0 0 0 0 1 2 1

let's define d'(x,y) as the sum of the squared differences of the
convolutions x' and y':
    d'(A,B) = d(A',B') = sum(1 1 1 1 0 0 0 0) = 4
    d'(B,C) =            sum(0 1 2 1 0 1 2 1) = 8
    d'(A,C) =            sum(1 2 1 0 0 1 2 1) = 8

d'(A,B) < d'(B,C) and d'(A,B) < d'(A,C) but d'(A,C) = d'(B,C).

So, we have partially succeeded by introducing a gradient when the
differences are small, but when the distance is greater, it "maxes out"
and there is no more gradient.  It would appear we would need a larger
kernel.  Let's try

[ 1 2 3 4 5 4 3 2 1 ]

again with zero padding.

A'' = 4 5 4 3 2 1 0 0
B'' = 3 4 5 4 3 2 1 0
C'' = 0 0 1 2 3 4 5 4

then:

d''(A,B) = sumsq( 1 1 1 1 1 1 1 0 ) = 7
d''(B,C) = sumsq( 3 4 4 2 0 2 4 4 ) = sum( 9 16 16 4 0 4 16 16 ) = 81
d''(A,C) = 102

success.
"""

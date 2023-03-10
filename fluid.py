#!/usr/bin/env python3
import sys,os,json


if 'MODEL_PARAMS' in os.environ:
    MP = json.load(open(os.environ['MODEL_PARAMS'],"r"))
else:
    MP = []
def mp(pn,dflt):
    if pn in MP:
        return MP[pn]
    else:
        return dflt


TRAINING=not ('-t' in sys.argv or '-tr' in sys.argv)
RESUMING='-r' in sys.argv or '-tr' in sys.argv
BATCHSIZE=mp('BATCHSIZE',1)
num_epochs = mp('NUM_EPOCHS',4000)
num_meta_epochs = mp('NUM_META_EPOCS',1)
ITERS=mp('ITERS',10)
SCRAMBLE_SIZE=175
import cv2
os.environ['CUDA_LAUNCH_BLOCKING']='1'
#import torch_dct as dct
import numpy as np
import matplotlib.pyplot as plt
import traceback
import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"COMPUTING DEVICE: {device}")

#from torchvision import datasets
#from torchvision.transforms import ToTensor
#from torchinfo import summary
import torch.nn as nn
import time
from glob2 import glob
timestamp = time.strftime("%Y%m%d_%H%M%S")
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import fftconvolve




import numpy as np
import torch
import torch.nn as nn

# PyTorch 1.6.0 and older versions
def dct1_rfft_impl(x):
    return torch.rfft(x, 1)

def dct_fft_impl(v):
    return torch.rfft(v, 1, onesided=False)

def idct_irfft_impl(V):
    return torch.irfft(V, 1, onesided=False)



def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)





dgc=-1
def datagenerator():
    global dgc
    dgc+=1
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

        if len(flowqueue) == ITERS:

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

k = datagenerator()


def TensorFluidUnit(k):
    fm_train,fm_target = next(k)
    fm_train2 = torch.cuda.FloatTensor(fm_train)
    fm_target2= torch.cuda.FloatTensor(fm_target)
    return fm_train2, fm_target2

def TensorFluidGen(k):
    while True:
        tensorbatch = []
        while len(tensorbatch) < SCRAMBLE_SIZE:
            tensorbatch.append(TensorFluidUnit(k))
        scri = np.random.permutation(len(tensorbatch))
        for i in scri:
            yield tensorbatch[i]

tfgen = TensorFluidGen(k)

def TensorFluid(tfgen):
    return next(tfgen)



if TRAINING:
    datadir = f"datadir{os.path.sep}{timestamp}"
    os.makedirs(datadir, exist_ok=True)
    print(f"TRAINING: datadir={datadir}")

if (not TRAINING) or RESUMING:
    sep=os.path.sep
    resdatadir= sep.join(sorted(glob(f"datadir{sep}*{sep}cnnstate.dat"))[-1].split(sep)[:-1])
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
        emd0 = torch.cuda.FloatTensor((d0[assignment0].sum() / np.prod(x.shape),))
        emd1 = torch.cuda.FloatTensor((d1[assignment1].sum() / np.prod(x.shape),))
        emd2 = torch.cuda.FloatTensor((d2[assignment2].sum() / np.prod(x.shape),))
        return (emd0+emd1+emd2*0.1)*100.0

class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss,self).__init__()

    def forward(self,y,x):
        return (torch.exp(((y[:,0:3,...]-x[:,0:3,...]).abs()).sum()/np.prod(y.shape[0:-2])/100000.0)-1.0)*50

class ICNN(nn.Module):
    def conv(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=(4,4),leak=0.0,pool_size=1,dropout=0.0):
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
            out_channels=mp('NCHAN0',8),
            kernel_size=mp('KSIZE0',9),
            leak=mp('LEAK0',0.0),
            dropout=mp('DROPOUT0',0.0))

        convs_list = nn.ModuleList()
        for i in range(1,mp('NLAYERS',2)):
            convs_list.append(ICNN.conv(
                in_channels=mp(f'NCHAN{i-1}',8),
                out_channels=mp(f'NCHAN{i}',8),
                kernel_size=mp(f'KSIZE{i}',9),
                leak=mp(f'LEAK{i}',0.0),
                dropout=mp(f'DROPOUT{i}',0.0)))
        self.convs = nn.Sequential(*convs_list)

        self.convLast = ICNN.conv(
            in_channels=mp('NCHAN_LAST',8),
            out_channels=1,
            kernel_size=mp("KSIZE_LAST",9),
            leak=mp('LEAK_LAST',0.0),
            dropout=mp('DROPOUT',0.0))



    def forward(self,x):
        x1 = self.conv0(x)
        x1 = self.convs(x1)
        x1 = self.convLast(x1)

        return x1



cnn = ICNN()
#print(cnn)
#summary(cnn.cuda(), (1,1,175,256))
loss_func = nn.MSELoss()


from torch import optim
optlr = 0.001
optimizer = optim.Adam(cnn.parameters(), lr = optlr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.1,patience=5)

from torch.autograd import Variable

def norm(x):
    #x = np.abs(np.fft.ifft2(xp))
    return (x-x.min())/(x.max()-x.min())




def train(num_epochs, cnn, k):
    cnn.cuda()
    try:
        for me in range(num_meta_epochs):
            # Train the model
            total_step = 1 #traindata.shape[0] // BATCHSIZE
            #print(f"total_step = {total_step}")
            #optimizer.zero_grad()           
            for epoch in range(num_epochs):
                traindata,targetdata = TensorFluid(tfgen)
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
                    # gives batch data, normalize x when iterate train_loader
                    #b_x = Variable(images)   # batch x
                    #b_y = Variable(labels)   # batch y
                    #print(f"b_x.shape = {b_x.shape}")
                    output = cnn(traindata)
                    if ITERS > 1:
                        nextput = (cnn((output.to(device))))
                        output = nextput
                    loss = loss_func(output,targetdata[:,0:4,...].to(device))
                    #loss.requires_grad = True
                    #print(f"traindata.shape={traindata.shape}, output.shape={output.shape}", end='')
                    # clear gradients for this training step   
                    #optimizer.zero_grad()           
                    
                    # backpropagation, compute gradients 
                    loss.backward()
                    # apply gradients             
                    optimizer.step()
                    
                    if (i+1) % 1 == 0:
                        print ('{},{},{},{},{:.20f}' 
                               .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()),file=open(f'{datadir}/report.txt','a'))
                        print (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], tr/tg:{traindata[0,0,20,20]}/{targetdata[0,0,20,20]} log Loss: {np.log(loss.item()):.6f}')
                    pass

                if epoch % 50 == 0:
                    torch.save(cnn.state_dict(), f"{datadir}{os.path.sep}cnnstate.dat")
                    print(f"saved state to {datadir}{os.path.sep}cnnstate.dat",flush=True)
                
                scheduler.step(loss)
                #if loss.item() < 0.0002: break

    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)



if (not TRAINING) or RESUMING:
    cnn.load_state_dict(torch.load(f"{resdatadir}{os.path.sep}cnnstate.dat"))
    cnn.cuda()
    print(f"loaded {resdatadir}{os.path.sep}cnnstate.dat")

cnn.cuda()
cnn.train().cuda()
if TRAINING:
    print("training.")
    train(num_epochs, cnn, k)
    torch.save(cnn.state_dict(), f"{datadir}{os.path.sep}cnnstate.dat")
    print(f"saved state to {datadir}{os.path.sep}cnnstate.dat")


ktest = datagenerator()
tfgentest = TensorFluidGen(ktest)


def test():
    global test_data
    # Test the model
    cnn.eval()
    cnn.cuda()
    while(True):
        with torch.no_grad():
            test_data,test_target = TensorFluid(tfgentest)
            test_data = test_data.cuda()
            test_target = test_target.cuda()
            output = (cnn((test_data)))
            vorticity = output[0,0,...]
            vorticity = vorticity.cpu().detach().numpy()
            vort=np.zeros(vorticity.shape+(3,))
            vort[...,0] = np.abs(vorticity) * (vorticity>=0)
            vort[...,1] = np.abs(vorticity) * (vorticity>=0)
            vort[...,2] = np.abs(vorticity) * (vorticity<0)
            vort=norm(vort)

            cv2.imshow('test2',vort)
            ky = cv2.waitKey(5000)
            if ky in [3,27,81,113]: break
        pass
    pass



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

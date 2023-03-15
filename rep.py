#!/usr/bin/env python3
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from glob2 import glob
from scipy.signal import savgol_filter


def kalman(y):
    navg = 20
    bestloss=100
    beststd=100
    beststdepoch=None
    bestlossepoch=None
    yk = np.zeros_like(y)
    ys = np.zeros_like(y)
    for i in range(len(y)):
        yk[i]=y[max(0,i-navg):i+1].mean()
        ys[i]=y[max(0,i-navg):i+1].std()
        if yk[i]<bestloss and i >= 20:
            bestloss = yk[i]
            bestlossepoch = i
        if ys[i] < beststd and i >= 20:
            beststd = ys[i]
            beststdepoch = i
    if bestlossepoch is None:
        bestlossepoch=len(y)-1
        bestloss=yk[-1]
    if beststdepoch is None:
        beststdepoch = len(y)-1
        beststd = yk[-1]
    return yk, bestlossepoch,bestloss, ys,beststdepoch,yk[beststdepoch]
    
def kalman0(y):
    k = .5
    yk = np.array(y.copy())
    for i in range(1,len(y)):
        yk[i]=yk[i-1]*(1-k) + y[i]*k
        k= k*0.95 + (.05)*0.05
    return yk

repno=-1
if len(sys.argv)>1:
    repno = int(sys.argv[-1])

plt.figure(figsize=(8,5))
cols = list(plt.rcParams['axes.prop_cycle'])

coli = 0
datadirs = sorted(glob("datadir"+os.path.sep+"*"+os.path.sep+"report.txt"))

repdirs=[]
for i in range(repno,0):
    try:
        reps = datadirs[i]
        repdirs.append(reps)
    except:
        continue
xcoord=0
for reps in repdirs:
    repcol = cols[coli % 8]['color']
    coli+=1
    print(f"repcol={repcol}")
    x=pd.read_csv(reps,header=None)
    yvals=np.log10(np.array(x.iloc[:,-1]))
    lbl=reps.split(os.path.sep)[-2]
    plt.plot(range(xcoord,xcoord+len(yvals)),yvals,label=lbl, color=repcol,alpha=0.5,lw=0.5)
    #ysmooth=savgol_filter(yvals,min(len(yvals)//10+2,115)//2*2+1,1)
    ysmooth,minx,miny,ystd,minstdx,minstdy=kalman(yvals) 
    plt.plot(range(xcoord,xcoord+len(ysmooth)),ysmooth, color=repcol)
    plt.plot(xcoord+len(ysmooth)-1,ysmooth[-1], '.',ms=9,color=repcol,alpha=1.0,lw=1.5)
    plt.text(xcoord+len(ysmooth)-1,ysmooth[-1], lbl[19:])
    
    print(f"minx={minx},miny={miny}")
    plt.plot(xcoord+minx,miny, 'o',ms=14, mfc='#ffffff00',mec=repcol+"ff")
    plt.plot(xcoord+minstdx,minstdy, 'x',ms=14, mfc='#ffffff00',mec=repcol+"ff")
    print(f"minstdx,y={minstdx},{minstdy}")
    if '-x' not in sys.argv:
        xcoord += len(ysmooth)

leg = plt.legend(loc='best',prop={'size':6}, ncol=3)

# set the linewidth of each legend object
if 'legend_handles' in dir(leg):
    lhandles = leg.legend_handles
else:
    lhandles = leg.legendHandles

for legobj in lhandles:
    legobj.set_linewidth(1.5)
    legobj.set_alpha(1.0)



imgfile=os.path.sep.join((reps.split(os.path.sep)[:-1]))+os.path.sep+"report.png"
print(f"imgfile={imgfile}")
plt.savefig(imgfile)
plt.savefig("rep.tmp.jpg")
os.replace("rep.tmp.jpg","rep.jpg")
plt.show()
#if os.path.sep=='/': 
#    os.system("sync;sync;eom "+imgfile)
#else:
#    os.system("start "+imgfile)

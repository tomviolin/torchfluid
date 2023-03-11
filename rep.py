#!/usr/bin/env python3
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from glob2 import glob
from scipy.signal import savgol_filter
repno=-1
if len(sys.argv)>1:
    repno = int(sys.argv[1])
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
    yvals=np.log(x.iloc[:,-1])
    plt.plot(range(xcoord,xcoord+len(yvals)),yvals,label=reps[-6:], color=repcol,alpha=0.8,lw=0.8)
    ysmooth=savgol_filter(yvals,min(len(yvals)//10+2,115)//2*2+1,1)
    plt.plot(range(xcoord,xcoord+len(ysmooth)),ysmooth, color=repcol)
    plt.plot(xcoord+len(ysmooth)-1,ysmooth[-1], '.',ms=9,color=repcol,alpha=1.0,lw=1.5)
    miny = np.min(yvals)
    minx = np.where(yvals == miny)[0]
    plt.plot(xcoord+minx,miny, 'o',ms=14, mfc='#ffffff00',mec=repcol+"ff")
    xcoord += len(ysmooth)

leg = plt.legend()

# set the linewidth of each legend object
for legobj in leg.legend_handles:
    legobj.set_linewidth(1.5)
    legobj.set_alpha(1.0)



imgfile=os.path.sep.join((reps.split(os.path.sep)[:-1]))+os.path.sep+"report.png"
print(f"imgfile={imgfile}")
plt.savefig(imgfile)
plt.show()
#if os.path.sep=='/': 
#    os.system("sync;sync;eom "+imgfile)
#else:
#    os.system("start "+imgfile)

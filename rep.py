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
for i in range(repno,0):
    try:
        reps = datadirs[i]
    except:
        continue
    repcol = cols[coli % 8]['color']
    coli+=1
    print(f"repcol={repcol}")
    x=pd.read_csv(reps,header=None)
    plt.plot(np.log(x.iloc[:,-1]+0.1),label=str(i)+": "+reps, color=repcol,alpha=0.4)
    plt.plot(savgol_filter(np.log(x.iloc[:,-1]+0.1), min(x.shape[0]//10,115)//2*2+1,1), color=repcol)

leg = plt.legend()

# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.5)
    legobj.set_alpha(1.0)



imgfile=os.path.sep.join((reps.split(os.path.sep)[:-1]))+os.path.sep+"report.png"
print(f"imgfile={imgfile}")
plt.savefig(imgfile)
if os.path.sep=='/': 
    os.system("sync;sync;eom "+imgfile)
else:
    os.system("start "+imgfile)

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob2 import glob
from scipy.signal import savgol_filter

import argparse
parser = argparse.ArgumentParser(description='rapt.py')
parser.add_argument('-e', '--evo', type=str, default='evo/current_evo/evo_*', help='evo to report. ignored if --prev > 0')
parser.add_argument('-p', '--prev', type=int, default=0, help='previous evolutions, 0=current, 1=most recent previous, etc.')
args = parser.parse_args()

if args.prev == 0:
    evo = glob(args.evo)[0]
else:
    evo = sorted(glob("evo/prev_evos/evo_*"))[-args.prev]


print(f"evo: {evo}")

gens = sorted(glob(f"{evo}/prev_gens/gen_*"))[500:]
plt.close('all')
plt.figure(figsize=(10,4))
xcoord=1
hiavgs=[]
loavgs=[]

initscores = pd.read_csv(f"{gens[0]}/scores.csv").to_numpy()[:,-2:]

initscoresdict = scoresdict = { initscores[q,-2]: initscores[q,-1] for q in range(initscores.shape[0]) }


print (initscoresdict)
#stop
for gen in gens:

    thesescores = pd.read_csv(f"{gen}/scores.csv").to_numpy()[:,-2:]

    filt=[]
    for i in range(thesescores.shape[0]):
        filt += [ thesescores[i,0] in initscoresdict , ]
    scores=np.array(thesescores[filt,-1],dtype='float64')
    print(scores)
    print("-=-===")
    if len(scores) == 0: break
    plt.scatter([xcoord]*scores.shape[0], np.log10(scores),s=0.5,marker=".")
    """
    loavg = np.median(scores[scores < 10**-2])
    hiavg = np.median(scores[scores >= 10**-2])
    hiavgs += [hiavg,]
    loavgs += [loavg,]
    """
    xcoord += 1
"""
plt.plot(np.arange(1,xcoord),np.log10(loavgs),'-',lw=0.5)
plt.plot(np.arange(1,xcoord),np.log10(hiavgs),'-',lw=0.5)

plt.plot(np.arange(1,xcoord),savgol_filter(np.log10(loavgs),51,2),'-',lw=1)
plt.plot(np.arange(1,xcoord),savgol_filter(np.log10(hiavgs),51,2),'-',lw=1)
"""
plt.show()



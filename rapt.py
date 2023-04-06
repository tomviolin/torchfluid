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
parser.add_argument('-g', '--gens', type=int, default=0, help='how many generation from the end to report; default=0=all')
args = parser.parse_args()

if args.prev == 0:
    evo = glob(args.evo)[0]
else:
    evo = sorted(glob("evo/prev_evos/evo_*"))[-args.prev]


print(f"evo: {evo}")

gens = sorted(glob(f"{evo}/prev_gens/gen_*"))[-args.gens:]

plt.figure(figsize=(10,4))
xcoord=1
hiavgs=[]
loavgs=[]
for gen in gens:
    scores = pd.read_csv(f"{gen}/scores.csv").score.to_numpy()
    plt.scatter([xcoord]*scores.shape[0], np.log10(scores),s=0.5,marker=".",color='black',alpha=0.7)
    loavg = np.median(scores[scores < 10**-2])
    hiavg = np.median(scores[scores >= 10**-2])
    hiavgs += [hiavg,]
    loavgs += [loavg,]
    xcoord += 1

plt.plot(np.arange(1,xcoord),np.log10(loavgs),'-',lw=0.5)
plt.plot(np.arange(1,xcoord),np.log10(hiavgs),'-',lw=0.5)

plt.plot(np.arange(1,xcoord),savgol_filter(np.log10(loavgs),min(21, len(loavgs)//2*2-1),2),'-',lw=1)
plt.plot(np.arange(1,xcoord),savgol_filter(np.log10(hiavgs),min(21, len(hiavgs)//2*2-1),2),'-',lw=1)

plt.show()



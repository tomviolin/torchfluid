#!/usr/bin/env python3
import os,sys
from glob2 import glob
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime as dt

timestamp = dt.now().strftime("%Y%m%d_%H%M%S.%f")

agentlist = json.load(open("datadir/agent_params.json",'r'))

for agent_id in agentlist:
    if 'COMPLETED' in agentlist[agent_id]:
        print(f"Agent {agent_id}: {agentparams['COMPLETED']}

# establish data directory
if os.path.exists("datadir"):
    os.makedirs("archived_data",exist_ok=True)
    os.rename("datadir",f"archived_data{os.path.sep}archived_{timestamp}")
    os.makedirs("datadir", exist_ok=True)

# figure out where we are in the evolution process
agents = glob(f"datadir{os.path.sep}*{os.path.sep}report.txt")
if len(agents) > 30:
    print(f"agents found = {len(agents)}")
    # retrieve all agents' performance data
    agentscores = []
    agentparams = []
    for agentreportpath in agents:
        agent_params_path = os.path.dirname(agentreportpath)+f"{os.path.sep}model_params.json"
        if not os.path.exists(agent_params_path):
            continue
        try:
            agentreportdata = pd.read_csv(agentreportpath,header=None)
        except pd.errors.EmptyDataError as e:
            print("--empty--")
            continue
        print(agentreportdata.shape)
        agentscores.append(agentreportdata.iloc[-1,-1])
        agentparams.append(json.load(open(os.path.dirname(agentreportpath)+f"{os.path.sep}model_params.json",'r')))

    print(f"usable agents={len(agentscores)}")
    # sort agents by score
    argindex = np.argsort(agentscores)
    agentscores = np.array(agentscores)[argindex]
    agentparams = np.array(agentparams)[argindex]
    # top 1/3 will survive
    nsurvivors = len(agentscores)
    nsurvivors = nsurvivors//3
    if nsurvivors < 2:
        nsurvivors = 2

     

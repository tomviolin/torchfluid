#!/usr/bin/env python3

# establish data directory



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
            agentreportdata = pd.read_csv(agentreportpath,header=None)
            print(agentreportdata.shape)
            agentscores.append(agentreportdata.iloc[-1,-1])
            agentparams.append(json.load(open(os.path.dirname(agentreportpath)+f"{os.path.sep}model_params.json",'r')))

        # sort agents by score
        argindex = np.argsort(agentscores)
        agentscores = agentscores[argindex]
        agentparams = agentparams[argindex]
        # top 1/3 will survive
        nsurvivors = len(agentscores)

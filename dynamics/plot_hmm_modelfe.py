import pickle
import numpy as np
import matplotlib.pyplot as plt 
#from matplotlib.pyplot import cm
from glob import glob

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"

#%% Fetch free energy of all runs
run_ids = dict()
modelfe = dict()
for k in [8, 10]:
    hmm_out_dir = f"{deri_dir}/hmm/{k}states"
    modelfe[f"{k} states"] = []
    run_ids[f"{k} states"] = []
    for run_id in range(1, 11):
        run_dir = glob( f"{hmm_out_dir}/run{run_id}/model/history.pkl" )
        if len(run_dir)>0:
            history = pickle.load(open(f"{hmm_out_dir}/run{run_id}/model/history.pkl", "rb"))
            modelfe[f"{k} states"].append( history["free_energy"] )
            run_ids[f"{k} states"].append( run_id )
        else:
            print(f"Failed to find the corresponding model/history.pkl file for {k} states run {run_id}.")

#%% Plot 
fig = plt.figure(figsize=(4, 3))
ax  = fig.gca()
cmap= plt.cm.get_cmap('Set2')
for i, key in enumerate(modelfe.keys()):
    ax.scatter(run_ids[key], modelfe[key], marker='o', label=key, color=cmap(i))
ax.legend(frameon=False, loc='best')
ax.set_title("model fits: all runs")
ax.set_ylabel( 'free energy' )
ax.set_xlabel( 'run' )
# %%

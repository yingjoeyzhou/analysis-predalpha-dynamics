import os
import mne
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

K = 8

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"
hmm_out_dir = f"{deri_dir}/hmm/{K}states"

#Colors from Zhou et al. (2021)
colors = dict(conservative=(123/255,50/255,148/255), #purple
              liberal=(0/255,136/255,55/255)) #green

#%% =========== sub-function ================
# GET BEST RUN
def get_best_run():
    best_fe = np.Inf
    for run_id in range(1, 11):
        history = pickle.load(open(f"{hmm_out_dir}/run{run_id}/model/history.pkl", "rb"))
        if history["free_energy"] < best_fe:
            best_run = run_id
            best_fe = history["free_energy"]
    print("Best run:", best_run)
    return best_run


#%% ==========================================
#
#    1. READ IN THE EPOCHS OF THE BEST RUN
#    
#    2. COMPUTE CONDITION MEANS 
#
# ============================================
run_id = get_best_run()
epo_dir = f"{hmm_out_dir}/run{run_id}/epochs"
epo_fnames = sorted( glob( f"{epo_dir}/*-epo.fif" ) )

# Define the conditions and the selection arguments
conditions={"Conservative": "primetype==2 & taskname=='main'",
            "Cons-Corr": "primetype==2 & taskname=='main' & correct==1",
            "Cons-Incorr": "primetype==2 & taskname=='main' & correct==0",
            "Cons-Hit": "primetype==2 & taskname=='main' & presence==1 & correct==1",
            "Cons-Miss": "primetype==2 & taskname=='main' & presence==1 & correct==0",
            "Cons-CR": "primetype==2 & taskname=='main' & presence==0 & correct==1",
            "Cons-FA": "primetype==2 & taskname=='main' & presence==0 & correct==0",
            "Liberal": "primetype==8 & taskname=='main'",
            "Lib-Corr": "primetype==8 & taskname=='main' & correct==1",
            "Lib-Incorr": "primetype==8 & taskname=='main' & correct==0",
            "Lib-Hit": "primetype==8 & taskname=='main' & presence==1 & correct==1",
            "Lib-Miss": "primetype==8 & taskname=='main' & presence==1 & correct==0",
            "Lib-CR": "primetype==8 & taskname=='main' & presence==0 & correct==1",
            "Lib-FA": "primetype==8 & taskname=='main' & presence==0 & correct==0",           
            }

# Data dictionary
data = { condname:[] for condname in conditions.keys() }
for epo_fname in epo_fnames:
    print(f"Working on {epo_fname}...")
    epochs = mne.read_epochs( epo_fname )
    picks = mne.pick_channels_regexp( epochs.info['ch_names'], 'alpha*')
    for condname, sel_arg in conditions.items():
        data[condname].append( epochs[sel_arg].average(picks=picks)._data )

# Re-structure the matrices so that data[condname] is (n_subj, n_states, n_samples)
data['times'] = epochs.times
for condname in data.keys():
    data[condname] = np.stack( data[condname], axis=0 )


#%% =========== sub-function for plotting ============
def plot_stc(axes, data, condname, condcolor='k'):
    nsubjects = data[condname].shape[0]
    cnt = 0
    for ir in range(0, np.shape( axes )[0]):
        for ic in range(0, np.shape( axes )[1]):
            if cnt < K: #counting of the states < number of states
                y = np.mean( data[condname][:,cnt,:], axis=0 )
                er= np.std( data[condname][:,cnt,:], axis=0 ) / np.sqrt( nsubjects )
                axes[ir,ic].plot( data['times'], y, label=condname, color=condcolor )
                axes[ir,ic].fill_between( data['times'], y-er, y+er, alpha=0.2, color=condcolor )
                cnt = cnt + 1

    return axes


#%% =============================================================
#
#    PAIR-WISE COMPARISON OF STATE TIME COURSE: CONS vs. LIB
#
#   fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
#   fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
# ===============================================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
plot_stc(axes, data, condname="Conservative", condcolor=colors["conservative"])
plot_stc(axes, data, condname="Liberal", condcolor=colors["liberal"])
for ir in range(0, np.shape( axes )[0]):
    for ic in range(0, np.shape( axes )[1]):
        k = ir*np.shape( axes )[1] + ic
        axes[ir,ic].set_title( f"state {k}")
        axes[ir,ic].legend( loc="best", frameon=False )
        axes[ir,ic].set_xlim( -1, 1.2 )
fig.tight_layout()
fig.savefig(f"stc_cons_vs_lib_bestrun_of_{K}states.png")


#%% =============================================================
#
#  PAIR-WISE COMPARISON OF STC: CONSERVATIVE CORR vs. INCORRECT
#
#   fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
#   fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
# ===============================================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
plot_stc(axes, data, condname="Cons-Corr", condcolor=colors["conservative"])
plot_stc(axes, data, condname="Cons-Incorr", condcolor=(0.5,0.5,0.5))
for ir in range(0, np.shape( axes )[0]):
    for ic in range(0, np.shape( axes )[1]):
        k = ir*np.shape( axes )[1] + ic
        axes[ir,ic].set_title( f"state {k}")
        axes[ir,ic].legend( loc="best", frameon=False )
        axes[ir,ic].set_xlim( -1, 1.2 )
fig.tight_layout()
fig.savefig(f"stc_cons_corr_vs_incorr_bestrun_of_{K}states.png")


#%% =============================================================
#
#  PAIR-WISE COMPARISON OF STC: LIBERAL CORR vs. INCORRECT
#
#   fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
#   fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
# ===============================================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
plot_stc(axes, data, condname="Lib-Corr", condcolor=colors["liberal"])
plot_stc(axes, data, condname="Lib-Incorr", condcolor=(0.5,0.5,0.5))
for ir in range(0, np.shape( axes )[0]):
    for ic in range(0, np.shape( axes )[1]):
        k = ir*np.shape( axes )[1] + ic
        axes[ir,ic].set_title( f"state {k}")
        axes[ir,ic].legend( loc="best", frameon=False )
        axes[ir,ic].set_xlim( -1, 1.2 )
fig.tight_layout()
fig.savefig(f"stc_lib_corr_vs_incorr_bestrun_of_{K}states.png")


# %%

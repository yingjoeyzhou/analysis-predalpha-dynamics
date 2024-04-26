"""Extract pre-stimulus alpha from networks and predict behavior.
"""

import pickle
import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from osl_dynamics.inference import modes

# Global directories
id = int(1)
code_dir = "/Volumes/ExtDisk/analysis_DondersData/3018041.02"
out_dir  = f"{code_dir}/dynamics/results/run{id:02d}"
raw_dir  = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
deri_dir = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

#%% ======== Sub-function ========
def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    from scipy.io import loadmat, matlab
    import numpy as np

    def _check_vars(d):
        return d

    def _todict(matobj):
        return d

    def _toarray(ndarray):
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


#%% ============== Sub-function =================
def read_trialinfo(subject, raw_dir=raw_dir):

    # load the .mat file
    beh_dir = f"{raw_dir}/{subject}/beh"
    datamat = load_mat( glob(f"{beh_dir}/{subject}_TaskMEG*.mat")[0] )
    ##exptmat = load_mat( glob(f"{beh_dir}/{subject}_paramMEG.mat")[0] )

    # the experiment
    n_blocks = np.size( datamat['mainTrl'] )
    n_primes = np.size( datamat['primeTrl'][0].resp )
    n_trials = np.size( datamat['mainTrl'][0].resp ) + n_primes

    # the behavioral data ad a pandas.Dataframe object
    cnt = int(0)
    dat = pd.DataFrame(columns=['iTrial','iBlock','trialtype','condition','presence','correct','rt'])
    for b in range(n_blocks):
        n_presence = sum( datamat['primeTrl'][b].type )
        if n_presence==8:
            curr_cond = 'Conservative'
        else:
            curr_cond = 'Liberal'

        # priming trials first    
        for i in range(n_primes):
            #print(i)
            dat.loc[cnt] = [i, b, curr_cond, 'prime',
                            datamat['primeTrl'][b].type[i], 
                            datamat['primeTrl'][b].resp[i].correct, 
                            datamat['primeTrl'][b].resp[i].rt]
            cnt = cnt + 1

        # main trials
        for i in range(n_primes, n_trials):
            #print(i-n_primes)
            dat.loc[cnt] = [i, b, curr_cond, 'main',
                            datamat['mainTrl'][b].type[i-n_primes], 
                            datamat['mainTrl'][b].resp[i-n_primes].correct, 
                            datamat['mainTrl'][b].resp[i-n_primes].rt]
            cnt = cnt + 1

    # add a column of subject's exact report (report "present" or "absent")
    ans = np.logical_or( np.logical_and( dat['correct']==1, dat['presence']==1 ), #hit
                        np.logical_and( dat['correct']==0, dat['presence']==0 ) ) #FA
    ans = pd.array(ans, dtype=pd.UInt8Dtype())
    dat.insert(loc=5, column="report", value=ans)

    return dat
     

#%% =============== Sub-function ==================
def define_epochs(stc_subj, parc_subj):
    '''Epochs with trial-labeling of a particular subject.
    
    INPUTS:
    stc_subj : state time course (continuous)
    parc_subj: parcel's time course (continuous)
    '''
    
    raw = modes.convert_to_mne_raw(stc_subj, parc_subj, n_embeddings=15)
    events = mne.find_events(raw, shortest_event=0.5)

    # given a subject, load trial labeling
    subject = p[p.find("Sub"):(p.find("Sub")+6) ]
    metadata= read_trialinfo(subject, raw_dir=raw_dir)

    # JY: hard-coding triggers of interest and epochs' timing, to match the original PredAlpha pipeline
    stimOn   = 20 
    prestim  = 2   
    poststim = 2.7 

    # make mne.Epochs
    epochs = mne.Epochs(raw, events=events[events[:,-1]==stimOn,:], 
                        tmin=-prestim, tmax=poststim, baseline=None, 
                        metadata=metadata, reject_by_annotation=True, verbose="INFO")

    return epochs.drop_bad()



#%% ============ sub-funciton ===========
def plot_stateprob_timecourses_pairs(gavg, tvec, condname1, condname2, stat_timewin=[-1,1], epochs_dict=None):
    
    threshold = 4.5

    n_states = np.shape( gavg[condname1].data )[0]

    if stat_timewin is not None: #do stats
        ibeg = min(range(len(tvec)), key=lambda i: abs(tvec[i] - stat_timewin[0]))
        iend = min(range(len(tvec)), key=lambda i: abs(tvec[i] - stat_timewin[-1]))

        assert( epochs_dict is not None )
        condition1 = []
        condition2 = []
        for epo1, epo2 in zip( epochs_dict[condname1], epochs_dict[condname2] ):
            condition1.append( epo1.data[:,ibeg:iend])
            condition2.append( epo2.data[:,ibeg:iend])
        
        condition1 = np.stack(condition1,axis=0)
        condition2 = np.stack(condition2,axis=0)

    fig, axes = plt.subplots(2,4, figsize=(20,10))
    for i_state in range(0, n_states):
        ridx = 0 if i_state<4 else 1
        cidx = i_state if i_state<4 else i_state-4
        axes[ridx,cidx].plot( tvec, gavg[condname1].data[i_state,:], label=condname1 )
        axes[ridx,cidx].plot( tvec, gavg[condname2].data[i_state,:], label=condname2 )
        axes[ridx,cidx].set_title(f"State {i_state}")
        axes[ridx,cidx].legend()
        axes[ridx,cidx].set_xlabel('Time (s)')
        axes[ridx,cidx].set_ylabel('State probability')

        if stat_timewin is not None: #do stats and plot
            
            cond1_data = condition1[:,i_state,:]
            cond2_data = condition2[:,i_state,:]

            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                cond1_data-cond2_data,
                n_permutations=1024,
                threshold=threshold,
                tail=0,
                n_jobs=None,
                out_type="mask",
            )

            for i_c, c in enumerate(clusters):
                c = c[0]
                print(cluster_p_values[i_c])
                if cluster_p_values[i_c] <= 0.05:
                    h = axes[ridx,cidx].axvspan(tvec[c.start], tvec[c.stop - 1], color="r", alpha=0.3)
                else:
                    axes[ridx,cidx].axvspan(tvec[c.start], tvec[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

    
    
    fig.tight_layout()

    return fig


#%% DO THE WORK
# ========== Time-series data ==========
# Get inferred state time course
alp = pickle.load(open(f"{out_dir}/inf_params/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcellation file
src_dir  = f"{deri_dir}/src"
parc_files = sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif"))

# Number of states obtained
n_states = np.shape(stc[0])[-1]

# Make epochs
epochs_all  = []
epochs_sel  = {'Liberal':"trialtype=='Liberal' and condition=='main'",
               'Conservative': "trialtype=='Conservative' and condition=='main'",              
               'Hit': "presence==1 and correct==1 and condition=='main'",
               'Miss': "presence==1 and correct==0 and condition=='main'",
               'CR': "presence==0 and correct==1 and condition=='main'",
               'FA': "presence==0 and correct==0 and condition=='main'",
               'Correct': "correct==1 and condition=='main'",
               'Incorrect': "correct==0 and condition=='main'"}
epochs_dict = { k:[] for k in epochs_sel.keys() }
for p, s in zip(parc_files, stc):
    epochs = define_epochs( s, p )
    epochs_all.append(epochs)
    for cond_name, sel_arg in epochs_sel.items():
        tmp = epochs[sel_arg]
        epochs_dict[cond_name].append( tmp.average(picks=range(0,n_states)) )


# ============ Take average across subjects and plot =============
tvec = epochs.times 
gavg = { k:[] for k in epochs_dict.keys() }    
for cond_name, evoked in epochs_dict.items():
    gavg[cond_name] = mne.grand_average( epochs_dict[cond_name] )


# ================== Plot: individual subjects =================
i_state = 3
fig, ax = plt.subplots(1,4,figsize=(20,5))
for i, iSub in enumerate([0,10,20,30]):
    ax[i].plot( tvec, epochs_dict['Conservative'][iSub].data[i_state,:], label='Conservative' )
    ax[i].plot( tvec, epochs_dict['Liberal'][iSub].data[i_state,:], label='Liberal' )
    ax[i].set_title( f"subject{iSub} state{i_state}" )
    ax[i].legend()
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('State probability')
fig.tight_layout()
fig.savefig( "figures/stateprob_timecourse_showindiv" )


# ================== Plot: group-level ====================
fig = plot_stateprob_timecourses_pairs(gavg, tvec, condname1='Conservative', condname2='Liberal', 
                                       stat_timewin=[-1,0.15], epochs_dict=epochs_dict)
fig.savefig( "figures/stateprob_timecourse_group_Lib_vs_Cons" )

fig = plot_stateprob_timecourses_pairs(gavg, tvec, condname1='Hit', condname2='Miss', 
                                       stat_timewin=[-1,0.15], epochs_dict=epochs_dict)
fig.savefig( "figures/stateprob_timecourse_group_Hit_vs_Miss" )

fig = plot_stateprob_timecourses_pairs(gavg, tvec, condname1='Correct', condname2='Incorrect',
                                       stat_timewin=[-1,0.15], epochs_dict=epochs_dict)
fig.savefig( "figures/stateprob_timecourse_group_Corr_vs_Incorr" )
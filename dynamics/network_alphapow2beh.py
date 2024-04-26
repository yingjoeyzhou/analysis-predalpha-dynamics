"""Extract pre-stimulus alpha from networks and predict behavior.
"""

import pickle
import mne
import os
import numpy as np
import pandas as pd
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


#%% =============== Sub-function =================
def define_alphe_peakfreqs(f, psd, i_state, alpha_freq=(8, 13)):

    # Import FOOOF: using it without `pip` to avoid messing up my conda environment
    from sys import path
    path.insert(0,"/Volumes/ExtDisk/UTIL/fooof-1.1.0") #JY: hard-coded
    from fooof import FOOOFGroup

    # Use fooof
    fg = FOOOFGroup()
    fg.fit( f, np.mean(psd[:,i_state,:,:],axis=-2) )

    # Get fitted peak freqs
    pkfreq = fg.get_params(name='peak_params', col='CF')

    # For each subject, check if there exists a peak within the alpha range
    n_subjects = np.shape(psd)[0]
    peakfreqs  = np.zeros((n_subjects,2))
    for isub in range(n_subjects):
        sel = pkfreq[:,-1]==isub
        cur = [ cf for cf in pkfreq[sel,0] if np.logical_and(cf<=alpha_freq[1],cf>=alpha_freq[0]) ]
        if len(cur)>0:
            peakfreqs[isub,0] = cur[0] #Take the lower one when multiple alpha peaks are found
            peakfreqs[isub,1] = isub
        else:
            peakfreqs[isub,0] = np.nan
            peakfreqs[isub,1] = isub    

    return peakfreqs


#%% ============ sub-funciton ===========
def prepare_data_for_glmm(i_time, alphapow, behavior, export=True, export_dir=deri_dir):

    timekeys = [ k for k in alphapow.keys() ]
    n_states = np.shape( alphapow[timekeys[i_time]][0] )[-1]

    data = pd.concat( behavior )

    for i_state in range(n_states):
        apow = [ pow[:,i_state] for pow in alphapow[timekeys[i_time]] ]
        apow = np.hstack(apow)
        data.insert(loc=0, column=f"alphapow_state{i_state}", value=apow)

    subj = [ [i]*np.shape(p)[0] for i, p in enumerate(alphapow[timekeys[i_time]]) ]
    subj = np.hstack(subj)
    data.insert(loc=0, column="subject", value=subj)

    if export:
         tname = int(float(timekeys[i_time][2:])*1000)
         if not os.path.exists(export_dir):
            os.makedirs(export_dir)
         data.to_csv(f"{export_dir}/prep4glmm_prestim{tname}ms.csv" )

    return data


#%% DO THE WORK
# =========== alpha-peak freq ============
# Get states PSD
f = np.load( f"{out_dir}/spectra/f.npy")
psd = np.load( f"{out_dir}/spectra/psd.npy")

n_states = np.shape(psd)[1]
peakfreq = []
for i_state in range(n_states):
    pfreq = define_alphe_peakfreqs(f, psd, i_state=i_state, alpha_freq=(8, 13))
    print(f"STATE {i_state}: Did not find a robust alpha peak in {sum(np.isnan(pfreq[:,0]))} subject..")
    peakfreq.append(pfreq)


# ========== Time-series data ==========
# Get inferred state time course
alp = pickle.load(open(f"{out_dir}/inf_params/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcellation file
src_dir  = f"{deri_dir}/src"
parc_files = sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif"))

# Make epochs
epochs_all = []
for p, s in zip(parc_files, stc):
    epochs = define_epochs( s, p )
    epochs_all.append(epochs)


# =============== Compute Trial-by-Trial pre-stim alphapow ===========
# Define how we compute alpha power in which time window    
tvec = np.linspace(-1.5, 0.5, 21)
twin = 0.5
fwin = 4

# Loop through subjects
alphapow = {f"t={np.round(t,decimals=3)}":[] for t in tvec}
behavior = []
timekeys = [k for k in alphapow.keys()]
for i, epochs in enumerate(epochs_all):

    epochs = epochs["condition == 'main'"]
    print( epochs )

    behavior.append( epochs.metadata )
    n_trials = np.shape(epochs.metadata)[0]
    print( f"Including {n_trials} trials" )

    # within each subject, loop through times of channels/states of interest
    powmat = np.ones((n_trials,n_states))
    for i_time, t in enumerate(tvec):
        x = epochs.copy().get_data(tmin=t-twin/2, tmax=t+twin/2)

        # loop through diff. states
        for i_state in range(n_states):
            alphafreq = peakfreq[i_state][i,0]
            print(alphafreq)
            s,f = mne.time_frequency.psd_array_multitaper( x[:,i_state,:], sfreq=epochs.info['sfreq'], 
                                                        fmin=alphafreq-fwin/2, fmax=alphafreq+fwin/2, verbose=False)
            powmat[:,i_state] = np.mean(s, axis=-1)
    
        # concatenate data
        alphapow[timekeys[i_time]].append( powmat )

    # show progress
    print(f"Done processing data of subject {i+1} out of {len(epochs_all)}.")


# =========== prepare for GLMM (will do this in matlab =========
for i_time, t in enumerate(tvec):
    prepare_data_for_glmm( i_time, alphapow, behavior, 
                          export=True, export_dir=f"{deri_dir}/glmm" )
import os
import mne
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from osl_dynamics.analysis import spectral

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
# ============================================
run_id = get_best_run()
epo_dir = f"{hmm_out_dir}/run{run_id}/epochs"

tvec = np.arange(-1, 0.04, 0.04)
twin = np.ones( tvec.shape ) * 0.5

parc_fnames = sorted(glob(f"{deri_dir}/src/*/sflip_parc-raw.fif"))

for parc_fname in parc_fnames:

    subject = os.path.dirname(parc_fnames[0])[-6:]
    print(f"Working on {subject}...")

    #Read in the state time course
    epo_fname = glob( f"{epo_dir}/{subject}-epo.fif" )[0]
    stc = mne.read_epochs( epo_fname )
    picks = mne.pick_channels_regexp( stc.info['ch_names'], 'alpha*')
    stc.pick(picks=picks)

    #Read in the source-level activity, epoch the raw data the same way as "stc"
    raw = mne.io.read_raw(parc_fname)
    meta_csv = glob( f"{raw_dir}/{subject}/beh/*metadata.csv" )[0]
    metadata = pd.read_csv( meta_csv )
    events = mne.find_events(raw, min_duration=1/raw.info['sfreq'], shortest_event=1, verbose=False)
    events = mne.pick_events( events, include=metadata['trig_stimon'][0] )
    epochs = mne.Epochs(
            raw,
            events,
            event_id=metadata['trig_stimon'][0],
            metadata=metadata,
            tmin=stc.times[0], #was -2.0 in the original preproc setting
            tmax=stc.times[-1], #was 2.7 in the original preproc setting
            verbose=False,
            preload=True,
            reject_by_annotation=True,
        )
    epochs.pick(picks=None, exclude=["UPPT001"])
    
    #Make sure the same trials are used
    """
    if epochs.metadata.equals( stc.metadata)==False:
        to_include = epochs.metadata.isin(stc.metadata.tIdx)
        epochs.metadata['to_include'] = to_include.tIdx
        epochs = epochs["to_include==True"]
        epochs.metadata.drop(['to_include'],axis=1)
        assert( epochs.metadata.equals( stc.metadata) )
    """



    #Compute pre-stim alpha for main trials
    for t, w in zip(tvec, twin):
        tmin, tmax = t-w/2, t+w/2
        data = epochs["taskname=='main'"].get_data( tmin=tmin, tmax=tmax) #data is (n_epochs, n_ch, n_samples)
        alpha = stc["taskname=='main'"].get_data( tmin=tmin, tmax=tmax ) #alpha is (n_epochs, n_state, n_samples)

        #note: when only one window is used, Welch's method is equivalent to using a hanning taper
        f, psd = spectral.welch_spectra(data=np.swapaxes(data,1,2),
                            sampling_frequency=int(epochs.info['sfreq']),
                            alpha=np.swapaxes(alpha,1,2),
                            window_length=data.shape[-1],
                            step_size=data.shape[-1]*2,
                            frequency_range=[9, 12],
                            calc_coh=False,
                            keepdims=True,
        )



    

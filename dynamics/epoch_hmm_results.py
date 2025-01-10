"""Epoch state time courses.

"""

import os
import mne
import pickle
import numpy as np
import pandas as pd
from glob import glob

from osl_dynamics.inference import modes


K = 8

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"
hmm_out_dir = f"{deri_dir}/hmm/{K}states"

#%% GET BEST RUN
def get_best_run():
    best_fe = np.Inf
    for run_id in range(1, 11):
        history = pickle.load(open(f"{hmm_out_dir}/run{run_id}/model/history.pkl", "rb"))
        if history["free_energy"] < best_fe:
            best_run = run_id
            best_fe = history["free_energy"]
    print("Best run:", best_run)
    return best_run

run_id = get_best_run()


#%% EPOCH DATA WITH THE BEST RUN
# Calculate state time course (Viterbi path) from state probabilities
alp = pickle.load(open(f"{hmm_out_dir}/run{run_id}/inf_params/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcel data files
parc_files = sorted(glob(f"{deri_dir}/src/*/sflip_parc-raw.fif"))

# Output (epochs) directory
epo_out_dir = f"{hmm_out_dir}/run{run_id}/epochs"
os.makedirs(epo_out_dir, exist_ok=True)

# Record bad and good subjects
good_subj = []
bad_subj  = []
for s, p in zip(stc, parc_files):

    # Create an MNE raw object
    raw = modes.convert_to_mne_raw(s, p, n_embeddings=15)

    # Find events
    events = mne.find_events(raw, min_duration=1/raw.info['sfreq'], shortest_event=1, verbose=False)

    # Read in the metadata
    subject  = os.path.dirname(p)[-6:]
    meta_csv = glob( f"{raw_dir}/{subject}/beh/*metadata.csv" )[0]
    metadata = pd.read_csv( meta_csv )

    # Stimulus on events
    events = mne.pick_events( events, include=metadata['trig_stimon'][0] )

    # Quick check
    if np.shape(metadata)[0] == np.shape(events)[0]:

        # Epoch
        epochs = mne.Epochs(
            raw,
            events,
            event_id=metadata['trig_stimon'][0],
            metadata=metadata,
            tmin=-2, #was -2.0 in the original preproc setting
            tmax=2.7, #was 2.7 in the original preproc setting
            verbose=False,
        )

        # Save
        filename = f"{epo_out_dir}/{subject}-epo.fif"
        epochs.save(filename, overwrite=True)

        # Show progress
        print(f"Finished epoching and saving data of {subject}..")
    
        good_subj.append( subject )

    else:
        print(f"Something went wrong when epoching data of {subject}..")

        bad_subj.append( subject )


#%% Final check
if len(bad_subj)==0:
    print("All good! No bad subjects!")

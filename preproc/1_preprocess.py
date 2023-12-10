"""Preprocess the CTF raw data from predalpha.
"""

import glob
import sys
import mne
from os import path
from osl import preprocessing, utils

#Deal with the input
print(sys.argv)
if sys.argv[0]=='':
    subject = input("Subject ID (e.g., Sub201):")
    #subject = 'Sub207";
else:
    subject = sys.argv[1]
    print(sys.argv[1])

#Directories and files
RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

#Subject-specific .ds folder (inputs)
SUBJ_DIR     = path.join(RAW_DIR, subject)
SUBJ_MEG_RAW = glob.glob( f"{SUBJ_DIR}/meg/{subject.lower()}_*.ds" )

#Subject-specific derivatives folder (outputs)
SUBJ_PREPROC_DIR = path.join(DERIV_DIR, subject, "preproc")
if path.isdir(SUBJ_PREPROC_DIR)==False:
    path.os.mkdir(SUBJ_PREPROC_DIR)

#%% ================== sub-function ===================
# Auto-reject with no specific ECG channels
def ica_autoreject_no_ecg(dataset, userargs):
    import numpy as np

    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0}".format("ICA Autoreject"))
    logger.info("userargs: {0}".format(str(userargs)))

    # User specified arguments and their defaults
    eogmeasure = userargs.pop("eogmeasure", "correlation")
    eogthreshold = userargs.pop("eogthreshold", 0.35)
    ###ecgmethod = userargs.pop("ecgmethod", "ctps")
    ###ecgthreshold = userargs.pop("ecgthreshold", "auto")
    remove_components = userargs.pop("apply", True)

    # Reject components based on the EOG channel
    eog_indices, eog_scores = dataset["ica"].find_bads_eog(
        dataset["raw"], threshold=eogthreshold, measure=eogmeasure,
    )
    dataset["ica"].exclude.extend(eog_indices)
    logger.info("Marking {0} as EOG ICs".format(len(eog_indices)))

    '''
    =================== from OSL ==================
    # Reject components based on the ECG channel
    ecg_indices, ecg_scores = dataset["ica"].find_bads_ecg(
        dataset["raw"], threshold=ecgthreshold, method=ecgmethod
    )
    =================== from OSL ==================
    '''

    # Reject components based on the ECG channel
    ch_mag = np.array(dataset["raw"].ch_names)[mne.pick_types(raw.info, meg="mag")]
    ecg_indices, ecg_scores = dataset["ica"].find_bads_ecg(
        dataset["raw"], ch_name=ch_mag, threshold='auto', method='ctps'
    )
    dataset["ica"].exclude.extend(ecg_indices)
    logger.info("Marking {0} as ECG ICs".format(len(ecg_indices)))

    # Remove the components from the data if requested
    if remove_components:
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")

    return dataset



#%% Run batch processing
utils.logger.set_up(level="INFO")

# Settings
'''
================================== NOTE TO MYSELF ======================================
ABOUT EOGs:
The EOG channels are Eyelink eye-tracker duplicates 
(see https://www.fieldtriptoolbox.org/getting_started/eyelink/#meg-data---uadc-channels)

ABOUT ECG:
We did not record ECGs during the MEG session.
================================== NOTE ENDS ===========================================
'''
config = """
    preproc:
    - set_channel_types: {'UADC005-4302':eog, "UADC006-4302":eog}
    - pick: {picks: [mag, eog]}
    - filter: {l_freq: 0.25, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 300}
    - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
    - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
    - bad_channels: {picks: mag, significance_level: 0.1}
    - ica_raw: {picks: mag, n_components: 0.99}
    - ica_autoreject_no_ecg: {picks: mag, eogmeasure: correlation, eogthreshold: auto}
    - interpolate_bads: {}
"""

# Run batch preprocessing
preprocessing.run_proc_batch(
    config,
    files=SUBJ_MEG_RAW,
    outnames=[f"{subject}_preproc_raw.fif"],
    outdir=SUBJ_PREPROC_DIR,
    overwrite=True,
    dask_client=False,
    extra_funcs=[ica_autoreject_no_ecg],
)
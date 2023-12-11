"""Preprocessing of the CTF data at sensor-level. 
        - preprocess (detect bads, downsample, and etc.)
        - run ICA (further clean up EOG- and ECG-related artefacts)
"""

import os
import glob
from osl import preprocessing, utils
from dask.distributed import Client

# Define subjects (data) to process
subjects = ['Sub203','Sub204']#,'Sub205','Sub206','Sub207'] #'Sub207'

# Define procedures to apply
proc_to_run = ['ica'] #'preprocess']#, 'ica']

# Directories
PROJ_DIR = "/Volumes/ExtDisk/DATA/3018041.02"
RAW_DIR  = f"{PROJ_DIR}/rawbids"
DERIV_DIR= f"{PROJ_DIR}/derivatives"

# Define output directories
PREPROC_DIR = os.path.join( DERIV_DIR, "preproc" )
CLEANED_DIR = os.path.join( DERIV_DIR, "cleaned" )

#%% Some sub-functions
#%% ================== sub-function ===================
# Define the in- and out- to run preprocessing
def define_preproc_input_and_output(subject, RAW_DIR=RAW_DIR, DERIV_DIR=DERIV_DIR, PREPROC_DIR=PREPROC_DIR):
    '''Given a subject, define the raw MEG file and the output filename for preprocessing.

    Parameter
    ---------
    subject: string, e.g., "Sub203"

    Returns
    -------
    SUBJ_MEG_RAW    : a string, the full filename to the .ds folder (CTF data).
    SUBJ_PREPROC_DIR: a string, the full filename to the output directory.
    SUBJ_PREPROC_RAW: a string, the .fif filename of the preproc output. 
    '''

    ####Directories and files
    ###RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
    ###DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

    #Subject-specific .ds folder (inputs)
    SUBJ_DIR     = os.path.join(RAW_DIR, subject)
    SUBJ_MEG_RAW = glob.glob( f"{SUBJ_DIR}/meg/{subject.lower()}_*.ds" )
    
    #Make sure both outputs are strings
    SUBJ_MEG_RAW = SUBJ_MEG_RAW[0]

    #Output file name in .fif format
    SUBJ_PREPROC_RAW = f"{subject}_preproc_raw.fif"

    #Subject-specific derivatives folder (outputs)
    SUBJ_PREPROC_DIR = os.path.join(PREPROC_DIR, os.path.splitext(SUBJ_PREPROC_RAW)[0])

    return SUBJ_MEG_RAW, SUBJ_PREPROC_RAW, SUBJ_PREPROC_DIR 

#%% ================== sub-function ===================
# Define the in- and out- to run ICA
def define_ICA_input_and_output(preproc_outfnames, preproc_outdirs ):
    '''Given a list of preproc output files, define inputs to and outputs from the ICA.
    '''
    SUBJ_ICA_INPUTS  = []
    ###SUBJ_ICA_OUTDIRS = []
    SUBJ_ICA_OUTPUTS = []
    for preproc_outfname, preproc_outdir in zip(preproc_outfnames, preproc_outdirs):
        SUBJ_ICA_INPUT = glob.glob( f"{preproc_outdir}/{preproc_outfname}" )
        ###SUBJ_ICA_OUTDIR= os.path.join( preproc_outdir, os.path.splitext(preproc_outfname)[0] )

        SUBJ_ICA_INPUTS.append( SUBJ_ICA_INPUT[0] )
        ###SUBJ_ICA_OUTDIRS.append( SUBJ_ICA_OUTDIR )
        SUBJ_ICA_OUTPUTS.append( preproc_outfname.replace('preproc','cleaned') )

    return SUBJ_ICA_INPUTS, SUBJ_ICA_OUTPUTS ###SUBJ_ICA_OUTDIRS

#%% ================== sub-function ===================
# Auto-reject with no specific ECG channels
def ica_autoreject_no_ecg(dataset, userargs):
    
    import numpy as np
    import mne
    import logging

    logger = logging.getLogger(__name__)

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
    ch_mag = np.array(dataset["raw"].ch_names)[mne.pick_types(dataset['raw'].info, meg="mag")]
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



#%% The inputs and outputs
infiles  = []
outfiles = []
outdirs  = []
for subject in subjects:
    
    '''
    === THIS WORKS BEST WHEN ONE SUBJECT HAS MULTIPLE RUNS OF RAW DATA ===
    #Subject-specific directories
    SUBJ_DERIV_DIR = f"{DERIV_DIR}/{subject}"
    if os.path.isdir( SUBJ_DERIV_DIR )==False:
        os.mkdir( SUBJ_DERIV_DIR )
    === THIS WORKS BEST WHEN ONE SUBJECT HAS MULTIPLE RUNS OF RAW DATA ===
    '''
       
    #Inputs and outputs: preprocessing
    in_meg_raw, outfname, out_dir = define_preproc_input_and_output(subject)
    infiles.append(in_meg_raw)
    outfiles.append(outfname)
    outdirs.append(out_dir)


#%% Do the work in batches
#===============================================
#
# 1. Run preprocessing
#
#===============================================
if "preprocess" in proc_to_run:

    #settings for preprocessing (up till ICA)
    config = """
        preproc:
        - set_channel_types: {'UADC005-4302':eog, "UADC006-4302":eog}
        - pick: {picks: [mag, eog]}
        - filter: {l_freq: 0.25, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
    """

    if __name__ == "__main__":

        utils.logger.set_up(level="INFO")

        client = Client(n_workers=2, threads_per_worker=1)

        #batch preprocessing
        preprocessing.run_proc_batch(
            config,
            files=infiles,
            outnames=outfiles,
            outdir=PREPROC_DIR,
            overwrite=True,
            dask_client=True,
        )


#===============================================
#
# 2. Run ICA on the preprocessed raw data
#
#===============================================
if "ica" in proc_to_run:

    #define fnames and directories
    infiles, outfiles = define_ICA_input_and_output( outfiles, outdirs )

    #settings for ICA
    config = """
        preproc:
        - ica_raw: {picks: mag, n_components: 0.99}
        - ica_autoreject_no_ecg: {picks: mag, eogmeasure: correlation, eogthreshold: auto}
        - interpolate_bads: {}
    """

    if __name__ == "__main__":

        utils.logger.set_up(level="INFO")
        
        client = Client(n_workers=2, threads_per_worker=1)

        #batch processing
        preprocessing.run_proc_batch(
            config,
            files=infiles,
            outnames=outfiles,
            outdir=CLEANED_DIR,
            overwrite=True,
            dask_client=True,
            extra_funcs=[ica_autoreject_no_ecg],
        )


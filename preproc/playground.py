import glob
from os import path
from dask.distributed import Client
from osl import preprocessing, utils


subject = "Sub207"

#Directories and files
RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

#Subject-specific .ds folder (inputs)
SUBJ_DIR     = path.join(RAW_DIR, subject)
SUBJ_MEG_RAW = glob.glob( f"{SUBJ_DIR}/meg/{subject.lower()}*.ds" )

#Subject-specific derivatives folder (outputs)
SUBJ_PREPROC_DIR = path.join(DERIV_DIR, subject, "preproc")
if path.isdir(SUBJ_PREPROC_DIR)==False:
    path.os.mkdir(SUBJ_PREPROC_DIR)

# Settings
# The EOG channels are Eyelink eye-tracker duplicates 
# (see https://www.fieldtriptoolbox.org/getting_started/eyelink/#meg-data---uadc-channels)
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
    - ica_autoreject: {picks: mag, ecgmethod: auto, eogthreshold: auto}
    - interpolate_bads: {}
"""

#%%
if __name__ == "__main__":

    utils.logger.set_up(level="INFO")

    # Setup parallel processing
    client = Client(n_workers=6, threads_per_worker=1)

    # Run batch preprocessing
    preprocessing.run_proc_batch(
        config,
        files=SUBJ_MEG_RAW,
        outdir=SUBJ_PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )


"""
#%%
from osl import source_recon

#Directories and files
RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

#Subject-specific .ds folder (inputs)
SUBJ_DIR         = path.join(RAW_DIR, subject)
SUBJ_PREPROC_DIR = path.join(DERIV_DIR, subject, "preproc")
SUBJ_MEG_CLEANED = glob.glob( f"{SUBJ_PREPROC_DIR}/{subject.lower()}_3018041" )
SUBJ_MRI_RAW     = glob.glob( f"{SUBJ_DIR}/anat/{subject}_T1w.nii.gz" )

#Subject-specific derivatives folder (outputs)
SUBJ_SRC_DIR = path.join(DERIV_DIR, subject, "src")
if path.isdir(SUBJ_SRC_DIR)==False:
    path.os.mkdir(SUBJ_SRC_DIR)

# Settings
config_coreg = """
    source_recon:  
    - compute_surfaces
        include_nose: true
    - coregister
        use_nose: true
        use_headshape: false
"""
source_recon.run_src_batch(
    config,
    src_dir=SUBJ_SRC_DIR,
    preproc_files=XXX,
    smri_files=SUBJ_MRI_RAW,

)



config_source_recon = """
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: mag
        rank: {mag: 120}
        parcellation_file: /Volumes/ExtDisk/UTIL/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

"""
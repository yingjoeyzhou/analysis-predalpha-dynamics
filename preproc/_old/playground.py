import glob
from os import path
from dask.distributed import Client
from osl import preprocessing, utils


subject = "Sub203"

#Directories and files
RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"



#%%
from osl import source_recon

#Directories and files
RAW_DIR   = "/Volumes/ExtDisk/DATA/3018041.02/rawbids"
DERIV_DIR = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"

#Subject-specific .ds folder (inputs)
SUBJ_MEG_CLEANED = glob.glob( f"{DERIV_DIR}/cleaned/{subject}_cleaned_raw/*cleaned_preproc_raw.fif" )
SUBJ_MRI_RAW     = glob.glob( f"{RAW_DIR}/{subject}/anat/{subject}_T1w.nii.gz" )

#Subject-specific derivatives folder (outputs)
SUBJ_SRC_DIR = path.join(DERIV_DIR, "src" )
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
    config_coreg,
    subjects=subject,
    src_dir=SUBJ_SRC_DIR,
    preproc_files=SUBJ_MEG_CLEANED,
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
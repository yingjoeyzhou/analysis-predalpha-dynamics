"""Preprocessing the data at source-level.
"""

import glob
import numpy as np
from os import path
from osl import source_recon, utils

'''
subjects = ['Sub203','Sub204','Sub205','Sub206','Sub207',
            'Sub208','Sub209','Sub210','Sub211','Sub212',
            'Sub214','Sub215','Sub216','Sub217','Sub220',
            'Sub221','Sub222','Sub223','Sub225','Sub226',
            'Sub227','Sub228','Sub229','Sub232','Sub234',
            'Sub238','Sub236','Sub239','Sub241','Sub243',
            'Sub244','Sub246']
'''
subjects = ['Sub203','Sub204','Sub205','Sub206','Sub207',
            'Sub208','Sub209','Sub210','Sub211','Sub212',
            'Sub214','Sub215','Sub216','Sub217','Sub220',
            'Sub221','Sub222','Sub223','Sub225','Sub226',
            'Sub227','Sub228','Sub229','Sub232','Sub234',
            'Sub238','Sub236','Sub239','Sub241','Sub243',
            'Sub244','Sub246']

proc_to_run = ["sign_flip"]

#Directories and files
PROJ_DIR  = "/ohba/pi/knobre/joeyzhou/3018041.02"
RAW_DIR   = f"{PROJ_DIR}/rawbids"
DERIV_DIR = f"{PROJ_DIR}/derivatives"
SRC_DIR   = f"{DERIV_DIR}/src"


#%% ============ sub-function ==============
def define_input_and_output( subject ):
    '''Define the fnames of inputs, given a subject.
    '''
    #Subject-specific .ds folder (inputs)
    SUBJ_MEG_CLEANED = glob.glob( f"{DERIV_DIR}/cleaned/{subject}_cleaned_raw/*cleaned_preproc_raw.fif" )
    SUBJ_MRI_RAW     = glob.glob( f"{RAW_DIR}/{subject}/anat/{subject}_T1w.nii.gz" )
    
    return SUBJ_MEG_CLEANED[0], SUBJ_MRI_RAW[0]


#%% ============== sub-function ===============
POS_FILE = RAW_DIR + "/{0}/anat/{0}_headshape.mat"
def save_polhemus_from_pos(src_dir, subject, preproc_file, smri_file, epoch_file):
    """Saves fiducials/headshape from a pos file."""

    from scipy.io import loadmat

    # Load pos file. Note, it contains values in cm in polhemus space
    pos_file = POS_FILE.format(subject)
    utils.logger.log_or_print(f"Saving polhemus from {pos_file}")
    data = loadmat( pos_file )

    # RHINO is going to work with distances in mm so we convert from cm to mm
    for v, k in enumerate(data):
        if 'pos' in k:
            data[k] = data[k] * 10

    polhemus_nasion = np.array( data['pos_nas'] ).astype("float64").T
    polhemus_rpa    = np.array( data['pos_rpa'] ).astype("float64").T
    polhemus_lpa    = np.array( data['pos_lpa'] ).astype("float64").T
    polhemus_headshape = np.array( data['pos'] ).astype("float64").T

    # Save
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)


#%% Do the work
preproc_files = []
smri_files = []
for subject in subjects:
    SUBJ_MEG_CLEANED, SUBJ_MRI_RAW = define_input_and_output( subject )
    preproc_files.append( SUBJ_MEG_CLEANED )
    smri_files.append( SUBJ_MRI_RAW )

#%% Step 1
if "coregister" in proc_to_run:

    print("Running coregistration...")

    # Settings
    config_coreg = """
        source_recon:  
        - save_polhemus_from_pos: {}
        - compute_surfaces:
            include_nose: true
        - coregister:
            use_nose: true
            use_headshape: false
    """

    # Run batch
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")
    source_recon.run_src_batch(
        config_coreg,
        subjects=subjects,
        src_dir=SRC_DIR,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[save_polhemus_from_pos],
    )

#%% Step 2
if "source_recon" in proc_to_run:
    config_src = """
        source_recon:
        - forward_model:
            model: Single Layer
        - beamform:
            freq_range: [1, 45]
            chantypes: mag
            rank: {mag: 120}
    """

    # Run batch
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")
    source_recon.run_src_batch(
        config_src,
        subjects=subjects,
        src_dir=SRC_DIR,
        preproc_files=preproc_files,
        smri_files=smri_files,
    )

#%% Step 3
if "parcellate" in proc_to_run:
    import os
    
    for subject in subjects:
        os.system(f"python3 preproc_util_parcellate.py '{subject}'")


#%% Step 4
if "sign_flip" in proc_to_run:

    utils.logger.set_up(level="INFO")

    # Find a good template subject to align other subjects to
    template = source_recon.find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    # Settings
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 3000
            max_flips: 20
    """

    # Run batch sign flipping
    source_recon.setup_fsl("/opt/ohba/fsl/6.0.5")
    source_recon.run_src_batch(config, SRC_DIR, subjects, dask_client=False)        
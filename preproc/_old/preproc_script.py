"""Script for preprocessing. 
"""

import os
import sys
import glob

# Define subjects (data) to process
subjects = ['Sub203'] #'Sub207'

# Define procedures to apply
proc_to_run = ['preprocess']

# Directories
PROJ_DIR = "/Volumes/ExtDisk/DATA/3018041.02"
RAW_DIR  = f"{PROJ_DIR}/rawbids"
DERIV_DIR= f"{PROJ_DIR}/derivatives"

# Loop through the subjects of interest
for subject in subjects:

    #Subject-specific directories
    SUBJ_DERIV_DIR = f"{DERIV_DIR}/{subject}"
    if os.path.isdir( SUBJ_DERIV_DIR )==False:
        os.mkdir( SUBJ_DERIV_DIR )
    
    #Run preprocessing when asked
    if "preprocess" in proc_to_run:
        os.system( f"python3 1_preprocess.py '{subject}'" )
        print(f"Done preprocessing on {subject}")
    
    #Run ICA on the preprocessed data
    if "ica" in proc_to_run:
        os.system()

    '''
    ==================== NOTE TO MYSELF =====================
    For HMM-related analysis, we won't epoch the data during 
    preprocessing. This ie because the HMM will give better 
    results the more training data we put in. 
    ==================== NOTE ENDS ==========================
    '''

    #Run coregistration when asked
    if "coreg" in proc_to_run:
        os.system( f"python3 2_coreg.py '{subject}'" )
        print(f"Done coregistration on {subject}")

    #Run source-reconstruction when asked
    if "source_recon" in proc_to_run:
        os.system( f"python3 3_source_recon.py '{subject}'" )
        print(f"Done source_recon on {subject}")
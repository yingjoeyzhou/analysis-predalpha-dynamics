import numpy as np
from osl_dynamics.config_api import wrappers

id = int(1)
code_dir  = "/Volumes/ExtDisk/analysis_DondersData/3018041.02/dynamics"
out_dir   = f"{code_dir}/results/run{id:02d}"
model_dir = out_dir + "/model"

# Load prepared data
load_data_kwargs = {'inputs': 'training_data/networks', 'prepare': {'tde_pca': {'n_embeddings': 15, 'n_pca_components': 80}, 'standardize': {}}, 'kwargs': {'store_dir': 'tmp_4297', 'sampling_frequency': 250, 'mask_file': 'MNI152_T1_8mm_brain.nii.gz', 'parcellation_file': 'Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz', 'n_jobs': 8}}
data = wrappers.load_data(**load_data_kwargs)

# 

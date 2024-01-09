from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm_networks.py 1")
    exit()
id = int(argv[1])

import os
from osl_dynamics import run_pipeline
from osl_dynamics.analysis import statistics, power, connectivity
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

#%% Directories
deri_dir = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"
src_dir  = f"{deri_dir}/src"

#%% Check if we have the training data prepared
if os.path.isdir('training_data/networks')==False:
    from prepare_data import save_data_networks
    save_data_networks(src_dir)

#%% DO THE WORK
# Configure the pipeline
config = """
    load_data:
        inputs: training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
            n_jobs: 8
        prepare:
            tde_pca: {n_embeddings: 15, n_pca_components: 80}
            standardize: {}
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 8
        nnmf_components: 2
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_hmm_network_summary_stats: {}
"""

# Run analysis
run_pipeline(config, output_dir=f"results/run{id:02d}")

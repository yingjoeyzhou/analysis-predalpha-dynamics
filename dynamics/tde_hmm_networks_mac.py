from sys import argv
from osl_dynamics import run_pipeline

if len(argv) != 3:
    print("Please pass n_states run_id, e.g. python tde_hmm_networks_mac.py 12 1")
    exit()
n_states_placeholder = argv[1]
run_id = argv[2]

#Directories
proj_dir = "/Volumes/ExtDisk/DATA/3018041.02"
deri_dir = f"{proj_dir}/derivatives"

from osl_dynamics.analysis import statistics, power, connectivity
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

#%% DO THE WORK
#Output dierectory
hmm_out_dir = f"{deri_dir}/hmm/{n_states_placeholder}states/run{run_id}"

#Configure the pipeline
config = """
    load_data:
        inputs: training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
            n_jobs: 4
        prepare:
            tde_pca: {n_embeddings: 15, n_pca_components: 100}
            standardize: {}
    train_hmm:
        config_kwargs:
            n_states: n_states_placeholder
            learn_means: False
            learn_covariances: True
            batch_size: 32
        save_inf_params: False
"""

"""
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

#Just to make sure the number of states is set correctly
config = config.replace('n_states_placeholder', str(n_states_placeholder))

# Run analysis
run_pipeline(config, output_dir=hmm_out_dir)


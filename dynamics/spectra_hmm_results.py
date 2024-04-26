'''Only plotting the spectra for the best run. 
'''
import os
import pickle
import numpy as np
from glob import glob
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

K = 10

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"
hmm_out_dir = f"{deri_dir}/hmm/{K}states"
hmm_dat_dir = f"{proj_dir}/analysis/hmm/training_data/networks"

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

#%% DATA OF INTEREST
# Directory for output
spectra_dir = f"{hmm_out_dir}/run{run_id}/spectra"
os.makedirs(spectra_dir, exist_ok=True)

# Infer parameters
inf_params_dir = f"{hmm_out_dir}/run{run_id}/inf_params"
alpha = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))

# Data for training
'''
array_files = sorted(glob(f"{hmm_dat_dir}/*.npy"))
data = [ np.load(array_file) for array_file in array_files ]
'''

# Parcel data files
parc_files = sorted(glob(f"{deri_dir}/src/*/sflip_parc-raw.fif"))
data = Data(
    parc_files,
    picks="misc",
    reject_by_annotation="omit",
    store_dir=f"tmp_{K}_run{run_id}",
    n_jobs=8,
)

# Trim time point we lost during time-delay embedding and separating
# the data into sequences
#
# Note:
# - n_embeddings must be the same as that used to prepare the training
#   data.
# - sequence_length must be the same as used in the Config to create
#   the model.
# The exact Config used can be found in `hmm/runX/model/config.yml`.
data_ = data.trim_time_series(n_embeddings=15, sequence_length=2000)


# Sanity check: the first axis should have the same numebr of time points
for x, a in zip(data_, alpha):
   assert(x.shape[0] == a.shape[0] )


#%% Calculate multitaper
# The resulting psd.shape is (n_states, n_channels, n_freqs)
#               coh.shape is (n_states, n_chan, n_chan, n_freqs)
f, psd, coh, w = spectral.multitaper_spectra(
    data=data_,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    return_weights=True,  # weighting for each subject when we average the spectra
    n_jobs=8,  # parallelisation, if you get a RuntimeError set to 1
)

np.save(f"{spectra_dir}/f.npy", f)
np.save(f"{spectra_dir}/psd.npy", psd)
np.save(f"{spectra_dir}/coh.npy", coh)
np.save(f"{spectra_dir}/w.npy", w)


#%% Calculate non-negative matrix factorisation (NNMF)

# We fit 2 'wideband' components
wb_comp = spectral.decompose_spectra(coh, n_components=2)

np.save(f"{spectra_dir}/nnmf_2.npy", wb_comp)

#%% Delete temporary directory
data.delete_dir()

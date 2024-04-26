from osl_dynamics.config_api import wrappers

code_dir = "/Volumes/ExtDisk/analysis_DondersData/3018041.02/dynamics"
id = int(1)
out_dir = f"{code_dir}/results/run{id:02d}"

# Load data
load_data_kwargs = {'inputs': 'training_data/networks', 'prepare': {'tde_pca': {'n_embeddings': 15, 'n_pca_components': 80}, 'standardize': {}}, 'kwargs': {'store_dir': 'tmp_4297', 'sampling_frequency': 250, 'mask_file': 'MNI152_T1_8mm_brain.nii.gz', 'parcellation_file': 'Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz', 'n_jobs': 8}}
data = wrappers.load_data(**load_data_kwargs)
print( np.shape(data) )

# Multitaper-spectra
wrappers.multitaper_spectra(data, output_dir=out_dir,
                            kwargs={'frequency_range': [1, 45], 'n_jobs': 1},
                            nnmf_components=2)



#%% Plotting
import numpy as np
import matplotlib.pyplot as plt
from osl_dynamics.utils import plotting

f = np.load( f"{out_dir}/spectra/f.npy")
psd = np.load( f"{out_dir}/spectra/psd.npy")
coh = np.load( f"{out_dir}/spectra/coh.npy")



#%% Plot PSD
# Calculate the group-average
gpsd = np.mean(psd, axis=0)

# Calculate the mean across channels and the standard error
p = np.mean(gpsd, axis=-2)
e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

# Plot group-average
n_states = gpsd.shape[0]
fig, ax = plotting.plot_line(
    [f] * n_states,
    p,
    errors=[p-e, p+e],
    labels=[f"State {i}" for i in range(n_states)],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    y_range=(0,0.125),
)
fig.savefig( f"tde_hmm_run{id:02d}_PSD_group_compressedYScale")


# Plot individual subjects
fig, axes = plt.subplots(2,4, figsize=(15,5))
n_subjects= np.shape(psd)[0]

p = np.mean(psd, axis=-2)

for i1 in range(4):
    print(i1)
    for iSub in range(n_subjects):
        axes[0,i1].plot( f, p[iSub,i1,:] )
    axes[0,i1].set_title(f"State {i1}")
for i2 in range(4,8):
    print(i2)
    for iSub in range(n_subjects):
        axes[1,i2-4].plot( f, p[iSub,i2,:] )
    axes[1,i2-4].set_title(f"State {i2}")    
fig.tight_layout()
fig.savefig( f"tde_hmm_run{id:02d}_PSD_allsubjects" )


#%% Plot brain activation-like map
from osl_dynamics.analysis import power

p = power.variance_from_spectra(f, psd)
print(p.shape)

# Calculate group average
gp = np.mean(p, axis=0)

# Plot (takes a few seconds to appear)
figs, axes = power.save(
    gp,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    subtract_mean=True,
    plot_kwargs={"cmap": "RdBu_r",
                 "darkneww": 0.4,
    }
)

for i, fig in enumerate(figs):
    fig.savefig( f"tde_hmm_run{id:02d}_topo_state{i}")

#%% Plot connectivity
from osl_dynamics.analysis import connectivity

# Calculate mean coherence
c = connectivity.mean_coherence_from_spectra(f, coh)
print(c.shape)

# Calculate the group average
gc = np.mean(c, axis=0)

# Threshold the top 2% of connections relative to the mean
gc_thres = connectivity.threshold(gc, percentile=98, subtract_mean=True)

# Plot
connectivity.save(
    gc_thres,
    filename=f'tde_hmm_run{id:02d}_network_',
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"
)

#%% Get state time courses
import pickle
from glob import glob
from osl_dynamics.inference import modes

# Directories
inf_params_dir = f"{out_dir}/inf_params"
plots_dir = f"{out_dir}/alphas"

# Get inferred state time course
alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
stc = modes.argmax_time_courses(alp)

# Parcellation file
deri_dir = "/Volumes/ExtDisk/DATA/3018041.02/derivatives"
src_dir  = f"{deri_dir}/src"
parc_files = sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif"))

p = parc_files[0]
s = stc[0]
raw = modes.convert_to_mne_raw(s, p, n_embeddings=15)
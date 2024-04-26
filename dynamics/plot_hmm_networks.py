import pickle
import numpy as np
from matplotlib.pyplot import cm
from osl_dynamics.utils import plotting
from osl_dynamics.analysis import power
from osl_dynamics.analysis import connectivity

K = 8

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"
hmm_out_dir = f"{deri_dir}/hmm/{K}states"
hmm_dat_dir = f"{proj_dir}/analysis/hmm/training_data/networks"

#Parcellation file used for parcellation and to-be-use for plotting
parcellation_file = f"{proj_dir}/analysis/preproc/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"


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


#%% Read spectra
spectra_dir = f"{hmm_out_dir}/run{run_id}/spectra"
f = np.load( f"{spectra_dir}/f.npy" )
psd = np.load( f"{spectra_dir}/psd.npy" )
coh = np.load( f"{spectra_dir}/coh.npy" )

#%% Plot PSD
# Calculate the group-average
gpsd = np.mean(psd, axis=0)

# Calculate the mean across channels and the standard error
p = np.mean(gpsd, axis=-2)
e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

# Plot
n_states = gpsd.shape[0]
assert( n_states==K )
fig, ax  = plotting.plot_line(
    [f] * n_states,
    p,
    errors=[p-e, p+e],
    labels=[f"State {i}" for i in range(n_states)],
    x_label="Frequency (Hz)",
    y_label="PSD (a.u.)",
    y_range=(0,0.125),
)
fig.savefig( f"figures/tde_hmm_{K}states_bestrun_PSD" )


#%% Plot brain activation-like map
p = power.variance_from_spectra(f, psd)
print(p.shape)

# Calculate group average
gp = np.mean(p, axis=0)

# Plot (takes a few seconds to appear)
figs, axes = power.save(
    gp,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file=parcellation_file,
    subtract_mean=True,
    plot_kwargs={"cmap": cm.coolwarm,
                 "darkneww": 0.4,
                 "symmetric_cbar": True,
                 "cbar_tick_format": "%.2f",
    }
)
for i, fig in enumerate(figs):
    ax = fig.gca()
    ax.set_title(f"state{i}")
    fig.savefig( f"figures/tde_hmm_{K}states_bestrun_powertopo_state{i}")


#%% Plot connectivity
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
    filename=f"figures/tde_hmm_{K}states_bestrun_network_",
    parcellation_file=parcellation_file,
    plot_kwargs={"edge_cmap": cm.plasma,
                 "edge_vmin": 0.05,
                 "edge_vmax": 0.15,
                 "display_mode": "lzr",
                },
)

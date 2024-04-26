import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fooof import FOOOF
from matplotlib.pyplot import cm

K = 8

#Directories
proj_dir = "/ohba/pi/knobre/joeyzhou/3018041.02"
raw_dir  = f"{proj_dir}/rawbids"
deri_dir = f"{proj_dir}/derivatives"
hmm_out_dir = f"{deri_dir}/hmm/{K}states"

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

# Subject-specific PSDs
nsub = psd.shape[0]
spsd = np.mean(psd, axis=-2)

# Apply fooof to individual subject
fm = FOOOF(min_peak_height=0.05, verbose=False)
df = dict(CF=[], state=[])
for k in range(0,K):
    statename = f"state {k}"
    for isub in range(0,nsub):
        fm.fit( f, np.squeeze(spsd[isub,k,:]) )
        cf = fm.get_params('peak_params', col='CF')
        cf = cf[ np.logical_and(cf<14, cf>7) ]
        df["CF"].append( cf[0] ) if len(cf)>0 else df["CF"].append( np.nan )
        df["state"].append( statename )
df = pd.DataFrame.from_dict( df )

# Calculate the group-average
gpsd = np.mean(psd, axis=0)

# Calculate the mean across channels and the standard error
p = np.mean(gpsd, axis=-2)
e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])


#%% Apply fooof to the group
fm = FOOOF(min_peak_height=0.05, verbose=False)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,6))
for ir in range(0, np.shape( axes )[0]):
    for ic in range(0, np.shape( axes )[1]):
        k = ir*np.shape( axes )[1] + ic
        print(k)
        fm.fit( f, p[k,:] )
        fm.plot( plot_peaks='shade-line', ax=axes[ir,ic], peak_kwargs=dict(color="green"))
        for item in ( axes[ir,ic].get_xticklabels() + axes[ir,ic].get_yticklabels()):
            item.set_fontsize(6)
        for txt in axes[ir,ic].get_legend().get_texts():
            txt.set_fontsize(6)
        for ax in ([axes[ir,ic].title, axes[ir,ic].xaxis.label, axes[ir,ic].yaxis.label]):
            ax.set_fontsize(8)        
        axes[ir,ic].get_legend().remove()
        axes[ir,ic].set_title( f"state {k}", fontsize=10 )
        axes[ir,ic].set_xticks( np.arange(0,50,4) )
        axes[ir,ic].set_xlim( 0.5, 45.5 )

fig.tight_layout(h_pad=3.0, w_pad=3.0)
fig.savefig(f"fooof_psd_{K}states_bestrun.png")




#%% Plot the summary statistics
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
if K<10:
    cmap = sns.color_palette(palette="tab10", n_colors=K)
else:
    cmap = sns.color_palette(palette="turbo", n_colors=K)
ax = sns.kdeplot(data=df, x="CF", hue="state", ax=axes[0], 
                 common_norm=False, palette=cmap, legend=False)
axes[0].set_xlabel("center frequency (Hz)")
axes[0].set_ylabel("probability")

sns.countplot(data=pd.DataFrame.dropna(df), x="state", hue="state", ax=axes[1], palette=cmap)
#axes[1].plot( [-5,K+5], np.ones((2,))*nsub*0.85, 'k:')
axes[1].set_ylim(1, nsub)
axes[1].set_yticks(np.arange(0,nsub+1,4))
axes[1].set_ylabel('number of subjects')
axes[1].set_xlim(-1, K)
axes[1].set_xticks(np.arange(0,K))
axes[1].set_xticklabels(np.arange(0,K))

fig.suptitle('Summary stats: alpha oscillatory components')
fig.tight_layout(w_pad=3.0)
fig.savefig(f"fooof_alphaosc_{K}states_bestrun.png")



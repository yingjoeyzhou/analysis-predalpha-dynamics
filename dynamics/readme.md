# Time-delayed embedding HMM on the PredAlpha MEG dataset
For a detailed description of the PredAlpha MEG study, refer to 
> Zhou, Y. J., Iemi, L., Schoffelen, JM., de Lange, F. P., & Haegens, S. (2021). Alpha oscillations shape sensory representation and perceptual sensitivity. Journal of Neuroscience, 41(46), 9581-9592.

## Prerequisites
* conda environment named `osld` where the osl-dynamics toolbox (https://github.com/OHBA-analysis/osl-dynamics/tree/main) is installed.
* conda environment named `fooof` where the fooof toolbox (https://fooof-tools.github.io/fooof/) is installed.
* Matlab R2020b, with the FieldTrip toolbox (https://www.fieldtriptoolbox.org/) installed.

## Steps to reproduce the results
Note it is in theory not possible to "reproduce" the exact TDE-HMM results due to the stochastic nature of training. But the resulting models are highly similar given a large amount of data (with some real structures).
### Train TDE-HMM models
1. Execute `prepare_data.py` to prepare the source-level parcellated sign-flipped continuous data for TDE-HMM model training.
2. Execute `tde_hmm_networks.py`, each may take 24 to 30 hours to complete, on a CPUs-only Linux computer. To train multiple in a serial manner, use `run_multiple_HMM.py`. 
3. Run `plot_hmm_modelfe.py` to visualize the free energy of each resulting HMM model.
### Analyze the resulting TDE-HMM models
1. Run `spectra_hmm_results.py` computes the spectra of each state/network and for each session/subject.
2. Run `epoch_hmm_vpath.py` and `epoch_hmm_stateprob.py` to epoch the state time courses (`stc`), either as Viterbi path (with zeros and ones) or as probabilities.
3. Run `compare_epochs.py` to compare `stc` between e.g., the conservertive vs. liberal conditions. 
4. To visualize the results
        - `plot_hmm_alpha_networks.py` (and `plot_hmm_networks.py`) shows the topomap and connectivity profiles of each alpha state/network (and the all-frequencies state/network). 
        - `plot_hmm_fooof_psd.py` shows the fooof-ed power spectral density plots for key nodes/areas/parcels identified within the alpha state/network.
5. Excute `compute_prestim_alpha.py` to compute network alpha power (the $P_t,_k$ term in the GLMM model linking alpha activity to perceptual decisions) for all trials. Use `run_multiple_compute_alpha.py` to specify the K-states models of interest and the kth state of interest. This step will generate multiple Matlab-friendly .csv files.
### Link network alpha activity to perceptual decisions (in Matlab)


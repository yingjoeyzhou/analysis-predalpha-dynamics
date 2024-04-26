# Train time-delayed embedding HMMs
0. `prepare_data.py` prepares the data for HMM
1. manual start with `tde_hmm_networks.py` or `tde_hmm_networks_mac.py`(this was not used in the end)
2. automated pipeline using `run_multiple_HMM.py`

# Summary statistics of the trained HMMs
0. `plot_hmm_modelfe.py` plots the free energy of all trained HMMs.

# Analyze the HMM outputs
by default, these analyses take only the best run corresponding to the K states HMM
1. `spectra_hmm_results.py` computes the PSD of each of the K states
   - `plot_hmm_fooof_psd.py` analyses the resulting group- and subject-level PSDs using fooof
   - `plot_hmm_networks.py` plots the resulting PSDs together with the activation-like topomap and connectivity map.
2. `epoch_hmm_results.py` segments the state time course into epochs/trials.
   - `compare_epochs.py` compares the state time courses across conditions

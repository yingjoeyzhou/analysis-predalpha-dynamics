# Preprocessing
These are the scripts to preprocessing CTF MEG data from the "predalpha" project.

## Prerequisites
- the `osl-py` toolbox installed
- some python IDE, e.g., spyder, to interact with figures
- the `FSL` toolbox 


## Usage
1. Run `preproc_sensorlevel.py` for all relevant subjects (N=32)
    - possible to use a dask client
    - the `artefact_reject` part requires python IDE.
2. Run `preproc_sourcelevel.py` for all relevant subjects
    - IMPORTANT to have `FSL` properly set up
    - the `parcellate` step was coded as `preproc_util_parcellate.py` to avoid memory shortage of the compters
  
## Note to myself
The sensor-level preprocessing was done on my MacBook pro 2021. And the source-level preprocessing was done on OHBA hbaws work stations, with `FSL` already properly installed.

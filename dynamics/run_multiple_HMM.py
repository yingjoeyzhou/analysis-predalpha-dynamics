import os
from os import system

for run_id in [7,8]:
    cmd = f"python tde_hmm_networks.py 10 {run_id}"
    print(cmd)
    system(cmd)
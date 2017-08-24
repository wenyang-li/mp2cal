#!/bin/bash

#SBATCH -J firstcal

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}

python /users/wl42/OmniCal/mp2cal/scripts/firstcal_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --outpath=./omni_sol/ --ubls=57_61,57_62 $obs
#python omni_run_multi.py -p $pol -C $poscal --ftype='uvfits' --iffits --omnipath='./' $obs

#!/bin/bash

#SBATCH -J omnical

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='fhd' --omnipath='./omni_fhd/' --ex_ubls='57_58' --tave --projdegen $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_new/' --wgt_cal --ex_ubls='57_58' --tave $obs
python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_new/' --wgt_cal --ex_ubls='57_58' --tave $obs

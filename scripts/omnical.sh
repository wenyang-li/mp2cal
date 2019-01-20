#!/bin/bash

#SBATCH -J omnical

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='fhd' --omnipath='./omni_fhd/' --ex_ubls='57_58' --tave --conv $obs
python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' --fhdpath='/users/wl42/data/wl42/FHD_out/fhd_Calibration_PhaseII/' --tave --ex_ubls=57_58,57_59 --conv $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' --omnipath='./omni_amp/' --fhdpath='/users/wl42/data/wl42/FHD_out/fhd_Calibration_PhaseII/' --tave --ex_ubls=57_58,57_59 --amp_only $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./chi_sol/' --wgt_cal --ex_ubls='57_58,57_59' $obs
#python omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_sol/' --wgt_cal --ex_ubls='57_58' --tave --conv $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_new/' --wgt_cal --ex_ubls='57_58' --tave $obs

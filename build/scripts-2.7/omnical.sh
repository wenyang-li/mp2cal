#!/bin/bash

#SBATCH -J omnical

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}
calpar=omni_sol/$obs.pp.fc.npz
python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_test/' --wgt_cal --ex_ubls='57_58' --projdegen $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_run_multi.py --divauto --tave --fitdegen -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_sol/' --calpar=$calpar --fhdpath='/users/wl42/data/wl42/FHD_out/fhd_PhaseII_Longrun_EoR0/' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_run_multi.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_fc/' --removedegen $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_run_multi.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./all_sol/' --calpar=$calpar --divauto --wgt_cal $obs

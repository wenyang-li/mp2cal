#!/bin/bash

#SBATCH -J INS

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='fhd' --omnipath='./omni_fhd/' --ex_ubls='57_58' --tave --conv $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_sol/' --wgt_cal --ex_ubls='57_58' --tave --conv $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./chi_sol/' --wgt_cal --ex_ubls='57_58' --tave $obs
#python omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_sol/' --wgt_cal --ex_ubls='57_58' --tave --conv $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_run_mwa.py -p 'xx,yy' -C $poscal --ftype='uvfits' --omnipath='./omni_new/' --wgt_cal --ex_ubls='57_58' --tave $obs
#python /users/wl42/OmniCal/MJW-MWA/Catalog_Run.py $obs'.uvfits' ./mwilensky/
#python plotflag.py $obs
#python applyauto.py $obs
python flagcenter.py $obs
#python /users/wl42/scripts/chisq.py $obs
#python cotter_fq_ave.py --rewrite --xpol --ins='./mwilensky/' $obs
#python /users/wl42/scripts/chisq.py -o '/users/wl42/data/wl42/OBS0/sol_chi/' -s '/users/wl42/data/wl42/OBS0/plot_chi/' $obs

#!/bin/bash

#SBATCH -J omniapply

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}

#python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_sol/' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --intype='uvfits' --appfhd --ave $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_fhd/' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --intype='fhd' --ave $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_sol/' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --npz='omni_sol/omniave' --intype='uvfits' --appfhd --projdegen $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_fhd/' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --intype='fhd' $obs
python /users/wl42/OmniCal/mp2cal/scripts/cal_apply_average.py --fhdpath='/users/wl42/data/wl42/FHD_out/fhd_Calibration_PhaseII/' --outpath='./' --subtract --omnipath='./calibration/' $obs
#python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_fhd/' --intype='fhd' --npz='/users/wl42/data/wl42/Nov2016EoR0/omni_fhd/omniave' $obs

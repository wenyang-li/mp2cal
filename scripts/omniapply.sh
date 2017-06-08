#!/bin/bash

#SBATCH -J omniapply

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}

#python /users/wl42/OmniCal/capo/wyl/scripts/omni_apply_multi.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_sol/%s.npz' --npz='/users/wl42/data/wl42/Nov2016EoR0/omni_sol/omniave' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --intype='uvfits' --flag_bls $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_apply_multi.py -p 'xx,yy' -C 'PhaseII_cal' --intype='fhd' --flag_bls --npz='/users/wl42/data/wl42/Nov2016EoR0/omni_fhd/omniave' --omnipath='./omni_fhd/%s.npz' $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_apply_multi.py -p 'xx,yy' -C 'PhaseII_cal' --intype='uvfits' --flag_bls --omnipath='./omni_fc/%s.npz' $obs
python /users/wl42/OmniCal/mp2cal/scripts/omni_apply_mwa.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_test/%s.npz' --metafits='/users/wl42/data/wl42/Nov2016EoR0/' --npz='/users/wl42/data/wl42/Nov2016EoR0/omni_test/omniave' --intype='uvfits' --appfhd $obs
#python /users/wl42/OmniCal/capo/wyl/scripts/omni_apply_multi.py -p 'xx,yy' -C 'PhaseII_cal' --omnipath='./omni_sol/%s.npz' --metafits='/users/wl42/Nov2016EoR0/' --intype='uvfits' --flag_bls --bpfit $obs

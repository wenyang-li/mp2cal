#!/bin/bash

#SBATCH -J QLTM

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}
python ssins_flag.py $obs

#module load ghostscript
#module load git/2.10.2
#module load imagemagick/7.0.7
#module load idl
#shopt -s expand_aliases; source $IDL/envi53/bin/envi_setup.bash

#idl -IDL_DEVICE ps -IDL_CPU_TPOOL_NTHREADS $ncores -e slurm_ps_job -args /users/wl42/data/wl42/FHD_out/fhd_int_PhaseII/ $obs

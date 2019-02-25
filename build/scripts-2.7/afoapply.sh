#!/bin/bash

#SBATCH -J omniapply

obsids=("$@")
obs=${obsids[$SLURM_ARRAY_TASK_ID]}

python cal_apply_average.py $obs

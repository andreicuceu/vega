#!/bin/bash
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -J info_sample3
#SBATCH -C cpu
#SBATCH -t 8:00:00

#OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export COBAYA_USE_FILE_LOCKING=False


cd /global/u2/w/wmturner/repos/vega/cobaya_interface

time srun -n 4 -c 4 --cpu_bind=cores cobaya-run -r info_sample3.yaml > /global/u2/w/wmturner/repos/vega/cobaya_interface/scripts/info_sample3.log 2>&1

wait
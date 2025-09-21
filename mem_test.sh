#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --mem=48G      
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=InteractionTest
#SBATCH --account=def-tpoisot

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
export TORCH_NCCL_ASYNC_HANDLING=1

srun python mem_test.py --cluster 
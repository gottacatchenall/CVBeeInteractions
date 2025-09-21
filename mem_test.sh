#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G      
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=InteractionTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
export TORCH_NCCL_ASYNC_HANDLING=1

srun python mem_test.py --cluster 
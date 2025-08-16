#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1  
#SBATCH --mem=16G      
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=SIMCLR_test

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt


export TORCH_NCCL_ASYNC_HANDLING=1

srun python SIMCLR.py --batch_size 128 --cluster --num_workers=1  --max_epochs 100 --species bees --temperature 1.0


#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2  
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G      
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=MultiGPUTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
export TORCH_NCCL_ASYNC_HANDLING=1

srun python 03_vit_test.py --batch_size 256 --cluster --max_epochs 100 --species bees --lr 1e-3 --num_workers 4 --persistent_workers --prefetch_factor 8 --model huge
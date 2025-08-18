#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:1  
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G      
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=DINOv3Test

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
export TORCH_NCCL_ASYNC_HANDLING=1

srun python 03_vit_test.py --batch_size 128 --cluster --max_epochs 100 --species plants --lr 5e-4 --num_workers 2 
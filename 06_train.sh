#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G      
#SBATCH --time=03:30:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=InteractionTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt
export TORCH_NCCL_ASYNC_HANDLING=1

srun python 06_interaction_training.py --batch_size 128 --cluster --max_epochs 100 --lr 3e-4 --num_workers 4 --model huge --min_crop_size 0.8
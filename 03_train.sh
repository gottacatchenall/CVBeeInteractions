#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2        
#SBATCH --tasks-per-node=2    
#SBATCH --cpus-per-task=1  
#SBATCH --mem=32G      
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ViTTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt


export TORCH_NCCL_ASYNC_HANDLING=1

srun python 03_vit_test.py --batch_size 512 --cluster --num_workers=1  --max_epochs 5 --species bees


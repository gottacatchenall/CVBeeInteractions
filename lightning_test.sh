#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=1  
#SBATCH --mem=16G      
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=LightningTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt


export TORCH_NCCL_ASYNC_HANDLING=1

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

export TORCH_LOGS="graph_breaks,recompiles"

srun python lightning.py  --batch_size 512 --cluster --num_workers=1  --max_epochs 15 --species bees


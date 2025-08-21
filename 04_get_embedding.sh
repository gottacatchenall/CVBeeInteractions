#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ExtractEmbeddingTest

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

export TORCH_NCCL_ASYNC_HANDLING=1

srun python 04_get_embedding.py --species bees --cluster --model huge --num_workers 4 --persistent_workers --prefetch_factor 8 


echo "Finished Bees."

srun python 04_get_embedding.py --species plants --cluster --model huge --num_workers 4 --persistent_workers --prefetch_factor 8 
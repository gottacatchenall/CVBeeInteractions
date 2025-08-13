#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --account=def-tpoisot
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=16G
#SBATCH --job-name=plant_embed_test
#SBATCH --output=EMBED_TEST_%x_$j_.out

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python model.py --cluster --nepoch 121 --model vit --lr 3e-4 --species plants
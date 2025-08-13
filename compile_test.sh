#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --account=def-tpoisot
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=8G
#SBATCH --job-name=TEST_compile
#SBATCH --output=%x_%j_.out

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

export TORCHINDUCTOR_CACHE_DIR=/project/def-tpoisot/mcatchen/TORCHINDUCTOR_CACHE

python compile_tests.py --cluster
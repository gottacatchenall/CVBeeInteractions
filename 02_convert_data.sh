#!/bin/bash
#SBATCH --account=def-tpoisot
#SBATCH --nodes 1             
#SBATCH --cpus-per-task=1  
#SBATCH --mem=16G      
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ConvertDataToBinary

module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python 02_convert_data_to_wds.py --cluster --species bees

echo "Bees done."

python 02_convert_data_to_wds.py --cluster --species plants
#!/bin/bash
#SBATCH -J metadrive_training
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

dos2unix run_main2.slurm

# Load anaconda
module load anaconda3/python3.8

# Load CUDA
module load cuda/11.4

# Source conda.sh to enable conda commands
source /public/software/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate autodrive

# Add the current directory to Python path so it can find the metadrive module
export PYTHONPATH=$PYTHONPATH:$HOME

# Run the Python script
python3 ./metadrive-main/main2.py

echo "Job finished at: $(date)"
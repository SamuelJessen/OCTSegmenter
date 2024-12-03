#!/bin/bash
#SBATCH --job-name=lumi_training
#SBATCH --account=project_465001544
#SBATCH --output=/projappl/project_465001544/OCTSegmenter/slurm_logs/slurm-%j.out
#SBATCH --error=/projappl/project_465001544/OCTSegmenter/slurm_logs/slurm-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=dev-g
#SBATCH --gpus-per-task=1

# Set MIOpen debug level to 6 for detailed logs
#export MIOPEN_LOG_LEVEL=6

# Run the job
srun /usr/bin/singularity exec -B /projappl/project_465001544/OCTSegmenter:/mnt -B /flash/project_465001544/:/data cotainrImage.sif python /mnt/lumi_training.py


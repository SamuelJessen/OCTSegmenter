#!/bin/bash
#SBATCH --job-name=lumi_training_gentuity_unet_cv
#SBATCH --account=project_465001544
#SBATCH --output=/projappl/project_465001544/OCTSegmenter/slurm_logs/slurm-%j.out
#SBATCH --error=/projappl/project_465001544/OCTSegmenter/slurm_logs/slurm-%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

# Add the OCTSegmenter directory to PYTHONPATH
export PYTHONPATH=/mnt:$PYTHONPATH

# Run the job
srun /usr/bin/singularity exec -B /projappl/project_465001544/OCTSegmenter:/mnt -B /flash/project_465001544/:/data cotainrImage.sif python /mnt/lumi_training/lumi_training_gentuity_unet_cv.py


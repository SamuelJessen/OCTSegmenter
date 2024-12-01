#!/bin/bash
#SBATCH --job-name=helloMnist
#SBATCH --account=project_465001544
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

# Change to the directory containing the script
cd /projappl/project_465001544

# Verify the current directory
pwd

# List the files to confirm hello.py is present
ls -l

# Run the job with Singularity
# Use -B to bind the directory so that the container can access the files correctly
srun /usr/bin/singularity exec -B /projappl/project_465001544/OCTSegmenter/scripts:/mnt -B /flash/project_465001544/:/data /projappl/project_465001544/OCTSegmenter/cotainrImage.sif python /mnt/hello.py

#!/bin/bash
#SBATCH --job-name=helloMnist
#SBATCH --account=project_465001544
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=dev-g
#SBATCH --gpus-per-task=1

# Change to the directory containing the script
cd /projappl/project_465001544

# Verify the current directory
pwd

# List the files to confirm hello.py is present
ls -l

# Add a debug statement to list files inside the container
srun /usr/bin/singularity exec -B /projappl/project_465001544:/mnt -B /flash/project_465001544/:/data cotainrImage.sif ls /mnt

ROOT_DIR="/data/data_terumo_smoke_test"
export ROOT_DIR
srun /usr/bin/singularity exec -B /projappl/project_465001544:/mnt -B /flash/project_465001544/:/data cotainrImage.sif python /mnt/training_loop.py --root_dir $ROOT_DIR


#!/bin/bash

# Check if a file path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Error: No SLURM job script provided."
    exit 1
fi

# Read the SLURM job script line by line
while IFS= read -r line
do
    # Extract values for specific SLURM options
    if [[ $line == *--ntasks=* ]]; then
        ntasks=$(echo $line | grep -oP '(?<=--ntasks=)\w+')
    elif [[ $line == *--ntasks=* ]]; then
        ntasks=$(echo $line | grep -oP '(?<=--ntasks=)\w+')
    elif [[ $line == *--gpus-per-task=* ]]; then
        gpus_per_task=$(echo $line | grep -oP '(?<=--gpus-per-task=)\w+')
    elif [[ $line == *--cpus-per-task=* ]]; then
        cpus_per_task=$(echo $line | grep -oP '(?<=--cpus-per-task=)\w+')
    elif [[ $line == *--mem=* ]]; then
        mem=$(echo $line | grep -oP '(?<=--mem=)\d+') 
    elif [[ $line == *--time=* ]]; then
        time=$(echo $line | grep -oP '(?<=--time=)\S+')
        hours=$(echo $time | cut -d":" -f1)
        minutes=$(echo $time | cut -d":" -f2)
        seconds=$(echo $time | cut -d":" -f3)
        runtime=$(echo "$hours + $minutes / 60 + $seconds / 3600" | bc -l)
    elif [[ $line == *--account=* ]]; then
        account=$(echo $line | grep -oP '(?<=--account=)\w+')
    fi
done < "$1"

# Check if any of the necessary directives are missing
for var in cpus_per_task mem gpus_per_task runtime account; do
    if [ -z "${!var}" ]; then
        echo "Error: Necessary SLURM directive missing: --$var"
        exit 1
    fi
done

# if 'Mem' is 0, then we're actually allocating the entire node:
if [ "$mem" -eq 0 ]; then
    mem=$(echo "$nodes * 512" | bc)
fi


# Calculating GPU-hours-billed (formula from https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/)
cpu_value=$(echo "$ntasks*$cpus_per_task / 8" | bc -l)
mem_value=$(echo "$mem / 64" | bc -l)
gcd_value=$(echo "$ntasks*$gpus_per_task " | bc -l)

max_value=$(echo -e "$cpu_value\n$mem_value\n$gcd_value" | sort -nr | head -n1)

echo "CPU value: $cpu_value"
echo "mem value: $mem_value"
echo "GPU value: $max_value"
echo "runtime: $runtime"

gpu_hours_billed=$(echo "$max_value * $runtime * 0.5" | bc -l)


# Extract allocation:
allocation=$(jq -r ".billing.gpu_hours.alloc" "/var/lib/project_info/users/$account/$account.json")

# Output the results and compare
echo "GPU hours billed by this job: $gpu_hours_billed"
echo "Total GPU hours allocated: $allocation"

if (( $(echo "$gpu_hours_billed > $allocation*0.1" | bc -l) )); then
    echo "Job would use more than 10% of total GPU allocation. Job does not start."
else
    echo "Job placed in queue."
    batch_job_id=$(sbatch "$1" | awk '{print $NF}') # Submit job and extract job ID
    echo "Batch job ID: $batch_job_id" # Print batch job ID
fi

exit 0

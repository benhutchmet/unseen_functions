#!/bin/bash
#SBATCH --job-name=sub-process-analogs
#SBATCH --partition=short-serial
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-process-analogs-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-process-analogs-%A_%a.err
#SBATCH --array=1960-2018

# Set up the usage messages
usage="Usage: ${init_month}"

# Check the number of CLI arguments
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the CLI arguments
init_month=$1

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_functions/unseen_analogs_functions.py"

# Echo the CLI arguments
echo "Init year; ${SLURM_ARRAY_TASK_ID}"
echo "Init month: ${init_month}"

# Run the script
#!/bin/bash
python ${process_script} \
    --init_year ${SLURM_ARRAY_TASK_ID} \
    --init_month ${init_month}

# End of file
echo "End of file"
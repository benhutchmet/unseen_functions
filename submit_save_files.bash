#!/bin/bash
#SBATCH --job-name=sub-save-files
#SBATCH --partition=high-mem
#SBATCH --mem=400000
#SBATCH --time=800:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-save-files-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-save-files-%A_%a.err

# Set up the usage message
usage="Usage: sbatch submit_save_files.bash ${first_year} ${last_year} ${model} ${variable} ${season} ${lead_year}"

# Check the number of CLI args
if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the args
first_year=$1
last_year=$2
model=$3
variable=$4
season=$5
lead_year=$6

# Echo the CLI arguments
echo "First Year: ${first_year}"
echo "Last Year: ${last_year}"
echo "Model: ${model}"
echo "Variable: ${variable}"
echo "Season: ${season}"
echo "Lead Year: ${lead_year}"

# set up the process script
process_script="/home/users/benhutch/unseen_functions/load_and_save_file.py"

module load jaspy

# Run the script
python ${process_script} \
    --first_year "${first_year}" \
    --last_year "${last_year}" \
    --model "${model}" \
    --variable "${variable}" \
    --season "${season}" \
    --lead_year "${lead_year}"
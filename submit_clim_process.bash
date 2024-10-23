#!/bin/bash
#SBATCH --job-name=sub-clim-process
#SBATCH --partition=high-mem
#SBATCH --mem=400000
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-clim-process-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-clim-process-%A_%a.err

# Set up the usage message
usage="Usage: sbatch submit_clim_process.bash ${variable} ${country} ${season} ${first_year} ${last_year} ${detrend} ${bias_correct}"

# Check the number of CLI args
if [ "$#" -ne 7 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the args
variable=$1
country=$2
season=$3
first_year=$4
last_year=$5
model_fcst_year="1"
lead_year="1-10"
detrend=$6
bias_correct=$7

# Echo the CLI arguments
echo "Variable: ${variable}"
echo "Country: ${country}"
echo "Season: ${season}"
echo "First Year: ${first_year}"
echo "Last Year: ${last_year}"
echo "model_fcst_year: ${model_fcst_year}"
echo "lead_year: ${lead_year}"
echo "Detrend: ${detrend}"
echo "Bias Correct: ${bias_correct}"

# set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_UNSEEN.py"

module load jaspy

# Run the script
python ${process_script} \
    --variable "${variable}" \
    --country "${country}" \
    --season "${season}" \
    --first_year "${first_year}" \
    --last_year "${last_year}" \
    --model_fcst_year "${model_fcst_year}" \
    --lead_year "${lead_year}" \
    --detrend "${detrend}" \
    --bias_correct "${bias_correct}"
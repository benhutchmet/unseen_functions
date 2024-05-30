#!/bin/bash
#SBATCH --job-name=sub-bias
#SBATCH --partition=short-serial
#SBATCH --mem=30000
#SBATCH --time=120:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit-bias-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit-bias-%A_%a.err

# Set up the usage messages
usage="Usage: ${start_year} ${end_year} ${lead_time} ${variable} ${obs_varname}"

# Check the number of CLI arguments
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the CLI arguments
start_year=$1
end_year=$2
lead_time=$3
variable=$4
obs_varname=$5

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_functions/bias_functions.py"

# Echo the CLI arguments
echo "strt_yr: ${start_year}"
echo "end_yr: ${end_year}"
echo "lead_time: ${lead_time}"
echo "variable: ${variable}"
echo "obs_varname: ${obs_varname}"
echo "process_script: ${process_script}"

# Run the script
python ${process_script} --variable ${variable} --obs_variable ${obs_varname} --lead_time ${lead_time} --start_year ${start_year} --end_year ${end_year}
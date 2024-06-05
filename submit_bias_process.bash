#!/bin/bash
#SBATCH --job-name=sub-bias
#SBATCH --partition=high-mem
#SBATCH --mem=30000
#SBATCH --time=1000:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit-bias-corr-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit-bias-corr-%A_%a.err

# Set up the usage messages
usage="Usage: ${model} ${variable} ${obs_variable} ${lead_time} ${start_year} ${end_year} ${experiment} ${frequency} ${month_bc}"

# Check the number of CLI arguments
if [ "$#" -ne 9 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the CLI arguments
model=$1
variable=$2
obs_varname=$3
lead_time=$4
start_year=$5
end_year=$6
experiment=$7
frequency=$8
month_bc=$9

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_functions/bias_functions.py"

# Echo the CLI arguments
echo "Model: ${model}"
echo "Variable: ${variable}"
echo "Obs Variable: ${obs_varname}"
echo "Lead Time: ${lead_time}"
echo "Start Year: ${start_year}"
echo "End Year: ${end_year}"
echo "Experiment: ${experiment}"
echo "Frequency: ${frequency}"
echo "Month BC: ${month_bc}"


# Run the script
#!/bin/bash
python ${process_script} \
    --model ${model} \
    --variable ${variable} \
    --obs_variable ${obs_varname} \
    --lead_time ${lead_time} \
    --start_year ${start_year} \
    --end_year ${end_year} \
    --experiment ${experiment} \
    --frequency ${frequency} \
    --engine "netcdf4" \
    --parallel False \
    --month_bc ${month_bc}
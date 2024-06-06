#!/bin/bash
#SBATCH --job-name=sub-form-df
#SBATCH --partition=short-serial
#SBATCH --time=1000:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit-form-df-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit-form-df-%A_%a.err

# Set up the usage messages
usage="Usage: ${model} ${variable} ${first_year} ${last_year} ${lead_time} ${country}"

# Check the number of CLI arguments
if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the CLI arguments
model=$1
variable=$2
first_year=$3
last_year=$4
lead_time=$5
country=$6

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_functions/process_model_demand.py"

# Echo the CLI arguments
echo "Model: ${model}"
echo "Variable: ${variable}"
echo "First Year: ${first_year}"
echo "Last Year: ${last_year}"
echo "Lead Time: ${lead_time}"
echo "Country: ${country}"


# Run the script
#!/bin/bash
python ${process_script} \
    --model "${model}" \
    --variable "${variable}" \
    --first_year "${first_year}" \
    --last_year "${last_year}" \
    --lead_time "${lead_time}" \
    --country "${country}" 
#!/bin/bash


path=${1}
job_directory=$(dirname $path)
script=$(basename $path)
name=${script%%.py}
output="${job_directory}/${name}.out"

echo "#!/bin/sh
#SBATCH --job-name=${script}
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=8
#SBATCH --tasks=1
#SBATCH -o ${output}
hostname
python ${path} ${2} ${3} ${4} ${5}
exit 0" > ${name}.job

sbatch ${name}.job




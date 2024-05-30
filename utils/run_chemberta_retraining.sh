#!/bin/bash


path=${1}
job_directory=$(dirname $path)
script=$(basename $path)
name=${script%%.py}
output="${job_directory}/${name}${2}${3}.out"

echo "#!/bin/sh
#SBATCH --job-name=${script}
#SBATCH --mem=40gb
#SBATCH --cpus-per-task=16
#SBATCH --tasks=1
#SBATCH -o ${output}
#SBATCH --exclude=node57,node40
hostname
python ${path} ${2} ${3} ${4} ${5} ${6} ${7} ${8}
exit 0" > ${name}.job

sbatch ${name}.job




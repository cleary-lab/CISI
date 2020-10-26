#! /bin/bash

#$ -cwd
#$ -N SpotFinder
#$ -t 197-200
#$ -q broad
#$ -P regevlab
#$ -l h_vmem=55g
#$ -l h_rt=24:00:00
#$ -e Logs/
#$ -o Logs/

sleep $((SGE_TASK_ID%60))
source /broad/software/scripts/useuse
source /home/unix/bcleary/.my.bashrc
conda activate cisi_test
export OMP_NUM_THREADS=1

OUT=../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages

read file tissue round channel <<< $(sed -n ${SGE_TASK_ID}p $OUT/tissue_channel_list.txt)

python spot_finder_starfish.py --path $file --out-path $OUT/$tissue/$round/$channel.tiff --spot-pad-pixels 2

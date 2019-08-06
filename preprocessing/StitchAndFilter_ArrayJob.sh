#! /bin/bash

#$ -cwd
#$ -N StitchAndFilter
#$ -t 1-56
#$ -q broad
#$ -P regevlab
#$ -l h_vmem=15g
#$ -l h_rt=3:00:00
#$ -e Logs/
#$ -o Logs/

sleep $((SGE_TASK_ID%60))
source /broad/software/scripts/useuse
source /home/unix/bcleary/.my.bashrc

#BASE=../../MotorCortex/20181016_MotorCortex_CompositeHCR
BASE=../../Kidney/20190304_Kidney_CompositeHCR

python stitch_from_fiji_coords.py --basepath $BASE/FilteredCompositeImages --channels conf-488,conf-561,conf-640,conf-405 --keep-existing --blank-round round6 --jobnum $((SGE_TASK_ID-1))


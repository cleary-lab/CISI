#! /bin/bash

#$ -cwd
#$ -N ParseND2
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

BASE=../../Kidney/20190304_Kidney_CompositeHCR
#BASE=../../MotorCortex/20181016_MotorCortex_CompositeHCR

python parse_and_z_project_nd2.py --inputdir $BASE/RawData --images-to-process $BASE/RawData/Images_to_Process.csv --rounds-to-channels $BASE/RawData/Rounds2Channels.csv --outdir $BASE --jobnum $((SGE_TASK_ID-1))

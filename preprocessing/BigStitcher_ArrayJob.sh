#! /bin/bash

#$ -cwd
#$ -N BigStitcher
#$ -t 1-62
#$ -q broad
#$ -P regevlab
#$ -l h_vmem=80g
#$ -l h_rt=48:00:00
#$ -e Logs/
#$ -o Logs/

sleep $((SGE_TASK_ID%60))
source /broad/software/scripts/useuse
source /home/unix/bcleary/.my.bashrc

BASE=../../MotorCortex/20200201_MotorCortex_HCR/RawData/
FUSE=/broad/hptmp/bcleary/BigStitcher/

files=($BASE/*.nd2)
file=$(basename ${files[$SGE_TASK_ID]} .nd2)

Fiji.app/ImageJ-linux64 --headless --console -macro BigStitcher_CreateFuseDataset.ijm "$BASE $file $FUSE"


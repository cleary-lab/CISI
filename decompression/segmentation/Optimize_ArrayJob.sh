#! /bin/bash

#$ -cwd
#$ -N Optimize
#$ -t 1-100
#$ -q broad
#$ -P regevlab
#$ -l h_vmem=20g
#$ -l h_rt=23:00:00
#$ -e Logs/
#$ -o Logs/

sleep $((SGE_TASK_ID%60))
source /broad/software/scripts/useuse
source /home/unix/bcleary/.my.bashrc
conda activate cisi_test

read cond corr <<< $(sed -n ${SGE_TASK_ID}p alpha_grid.txt)

python optimize_dictionary.py --basepath ../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/ --trainpath ../MotorCortex/Training/Simulations_38genes/ --region-mask Mask.tif --modules-in gene_modules.Allen.tpm.npy --modules-out optimized_modules/modules/gene_modules.array.${SGE_TASK_ID} --tissues 1,2,3,4,5,6,7,8,9,10,11,12 --batch-size 7500 --epochs 20 --method integrated_intensity --alpha-cond $cond --alpha-corr $corr

python decompress_segmented_cells.py --basepath ../MotorCortex/20200201_MotorCortex_HCR/ --trainpath ../MotorCortex/Training/Simulations_38genes/ --region-mask Mask.tif --modules optimized_modules/modules/gene_modules.array.${SGE_TASK_ID} --tissues 1,2,3,4,5,6,7,8,9,10,11,12 --use-test-idx --method integrated_intensity


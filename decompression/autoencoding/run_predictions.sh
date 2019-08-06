LOCAL=$PROJECT_DIR/FinalModels_"${MODEL_DATE}"
TF_RECORDS=$PROJECT_DIR/FilteredCompositeImages/ShuffledTFRecordBlocks/tfrecords_files
RECONSTRUCTED=$PROJECT_DIR/ReconstructedImages
START=0
END=23

mkdir $LOCAL/tmp_data
TRAIN_DATA=$LOCAL/tmp_data
cp data/*.npy $TRAIN_DATA/
mkdir $LOCAL/tmp_tiled
mkdir $LOCAL/tmp_tiled_encoded
OUTPUT_PATH=$LOCAL/tmp_tiled
for i in $(seq $START $END); do
	MODEL_NAME=final_"${MODEL_DATE}"_CISI_"${i}"
	MODEL_PATH=$LOCAL/$MODEL_NAME
	cp $(cat $TF_RECORDS".${i}.txt") $TRAIN_DATA/
	FOV_0=$(ls -l $TRAIN_DATA/fov_* | awk -F$'/' '{print $7}' | awk -F$'_' '{print $2}' | awk -F$'.' '{print $1}' | sort -k1,1n | head -n1)
	FOV_1=$(ls -l $TRAIN_DATA/fov_* | awk -F$'/' '{print $7}' | awk -F$'_' '{print $2}' | awk -F$'.' '{print $1}' | sort -k1,1n | tail -n1)
	echo $MODEL_PATH "$((FOV_0))-$((FOV_1+1))"
	python module/run_task.py \
		--job-dir $MODEL_PATH \
		--train-files $TRAIN_DATA \
		--eval-files $TRAIN_DATA \
		--fovs "$((FOV_0))-$((FOV_1+1))" \
		--num-layers 4 \
		--lambda-decode 5.95e-8 \
		--sparsity-k 3.43 \
		--lambda-tv 7.15e-7 \
		--lambdaW 0.01 \
		--num-filters 8 \
		--abundance-factor 18.66 \
		--lambda-abundance-factor 0.37 \
		--pixel-sparsity 0.06 \
		--loss-fn mse \
		--lambda-gene-correlation 0.011 \
		--predict-only \
		--predict-dir $OUTPUT_PATH
	for fp in $(cat $TF_RECORDS".${i}.txt"); do
		# super hacky...depending on how paths are set, 'tissue' and 'fov' might pull the wrong fields, so double check this
		tissue=$(echo $fp | awk -F$'/' '{print $11}')
		fov=$(echo $fp | awk -F$'/' '{print $13}' | awk -F$'_' '{print $2}' | awk -F$'.' '{print $1}')
		mv $OUTPUT_PATH/fov_"${fov}".* $RECONSTRUCTED/$tissue/tiled/
		mv "${OUTPUT_PATH}"_encoded/fov_"${fov}".* $RECONSTRUCTED/$tissue/tiled_encoded/
	done
	rm $TRAIN_DATA/fov_*
done



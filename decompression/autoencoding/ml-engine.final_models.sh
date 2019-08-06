# Run one job per tissue for final parameters
BUCKET_NAME="cisi-gcs-training"
REGION=us-east1
LOCAL=$PROJECT_DIR/FilteredCompositeImages/ShuffledTFRecordBlocks/tfrecords_files
START=0
END=23
# Make sure START and END match the range of ShuffledTFRecordBlocks/tfrecords_files.*.txt
DATA=data

gsutil mb -l $REGION gs://$BUCKET_NAME

for i in $(seq $START $END); do
	cp $(cat $LOCAL".${i}.txt") $DATA/
	TRAIN_DATA=gs://$BUCKET_NAME/$DATA"_${i}"
	EVAL_DATA=gs://$BUCKET_NAME/$DATA"_${i}"
	JOB_NAME=final_"$(date +%Y%m%d)"_CISI_"${i}"
	OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
	FOV_0=$(ls -l $DATA/fov_* | awk -F$'/' '{print $2}' | awk -F$'_' '{print $2}' | awk -F$'.' '{print $1}' | sort -k1,1n | head -n1)
	FOV_1=$(ls -l $DATA/fov_* | awk -F$'/' '{print $2}' | awk -F$'_' '{print $2}' | awk -F$'.' '{print $1}' | sort -k1,1n | tail -n1)
	echo $OUTPUT_PATH $TRAIN_DATA $EVAL_DATA "$((FOV_0))-$((FOV_1+1))"
	gsutil -m cp $DATA/* $TRAIN_DATA/
	gcloud ml-engine jobs submit training $JOB_NAME \
		--job-dir $OUTPUT_PATH \
		--runtime-version 1.12 \
		--python-version 3.5 \
		--module-name module.run_task \
		--package-path module/ \
		--region $REGION \
		--config config.yaml \
		-- \
		--train-files $TRAIN_DATA \
		--eval-files $EVAL_DATA \
		--fovs "$((FOV_0))-$((FOV_1+1))" \
		--train-steps 1000 \
		--decompress-steps 30000 \
		--num-layers 4 \
		--lambda-decode 1.16e-8 \
		--sparsity-k 2.5 \
		--lambda-tv 1.77e-8 \
		--lambdaW 0.026 \
		--num-filters 8 \
		--abundance-factor 16.2 \
		--lambda-abundance-factor 0.0085 \
		--pixel-sparsity 0.25 \
		--loss-fn l1 \
		--lambda-gene-correlation 8.9 \
		--eval-batch-size 16
	rm $DATA/fov_*
done

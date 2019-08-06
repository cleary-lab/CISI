BUCKET_NAME="cisi-gcs-training"
LOCAL=$PROJECT_DIR/FinalModels_"${MODEL_DATE}"
START=0
END=23

for i in $(seq $START $END); do
	MODEL_NAME=final_"${MODEL_DATE}"_CISI_"${i}"
	OUTPUT_PATH=$LOCAL/$MODEL_NAME
	mkdir $OUTPUT_PATH
	gsutil cp gs://$BUCKET_NAME/$MODEL_NAME/model.ckpt-30000.* $OUTPUT_PATH
	gsutil cp gs://$BUCKET_NAME/$MODEL_NAME/checkpoint $OUTPUT_PATH
	gsutil cp gs://$BUCKET_NAME/$MODEL_NAME/graph.pbtxt $OUTPUT_PATH
	echo $OUTPUT_PATH
done


# Run cloud hyperparameter tuning
BUCKET_NAME="cisi-gcs-hptuning"

REGION=us-east1

gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp -r data_mc gs://$BUCKET_NAME/

TRAIN_DATA=gs://$BUCKET_NAME/data
EVAL_DATA=gs://$BUCKET_NAME/data


JOB_NAME=cisi_hyperparameter_testing
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME


# hyperparameter tuning
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.12 \
		--python-version 3.5 \
    --module-name module.run_task \
    --package-path module/ \
    --region $REGION \
		--config hptuning_config.yaml \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
		--fovs 3-12 \
    --train-steps 300 \
		--decompress-steps 3000 \
		--num-layers 4 \
		--loss-fn l1



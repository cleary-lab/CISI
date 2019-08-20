python ../decompression/autoencoding/module/run_task.py \
	--job-dir model \
	--train-files data/ \
	--eval-files data/ \
	--fovs 27-29 \
	--train-steps 100 \
	--decompress-steps 1000 \
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
	--lambda-gene-correlation 8.9

mkdir output
mkdir output_encoded
python ../decompression/autoencoding/module/run_task.py \
	--job-dir model \
	--train-files data/ \
	--eval-files data/ \
	--fovs 27-29 \
	--num-layers 4 \
	--num-filters 8 \
	--predict-only \
	--predict-dir output


import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

try:
	from module import ccae
except:
	import ccae

NUM_GPUS=4

def _get_session_config_from_env_var():
	"""Returns a tf.ConfigProto instance that has appropriate device_filters set."""
	tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
	if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']):
		# Master should only communicate with itself and ps
		if tf_config['task']['type'] == 'master':
			return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
		# Worker should only communicate with itself and ps
		elif tf_config['task']['type'] == 'worker':
			return tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % tf_config['task']['index']])
	return None

def train_and_evaluate_combined(hparams):
	# Train the AutoEncoder
	fov_start = min(int(s) for s in hparams.fovs.split(','))
	FP_train = ['%s/fov_%s.tfrecords' % (hparams.train_files,fov) for fov in hparams.fovs.split(',')]
	train_input = lambda: get_dataset(FP_train, hparams.train_batch_size, epochs=hparams.num_epochs_decompression)
	train_spec = tf.estimator.TrainSpec(train_input, max_steps=hparams.decompress_steps)
	eval_input = lambda: get_dataset(FP_train, hparams.eval_batch_size)
	eval_spec = tf.estimator.EvalSpec(eval_input, steps=hparams.eval_steps)
	# To use multi-gpu need to wait for google folk to install nccl on training instances
	# see: https://stackoverflow.com/questions/52954413/gcmle-notfounderror-libnccl-so-2-cannot-open-shared-object-file
	# Once this works, might also want to prefetch, like here: https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
	#strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
	#run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var(), train_distribute=strategy)
	run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())
	run_config = run_config.replace(model_dir=hparams.job_dir)
	print('Model dir %s' % run_config.model_dir)
	estimator = tf.estimator.Estimator(model_fn=build_estimator_combined,params={'hparams': hparams, 'num_fov': len(FP_train), 'fov_start': fov_start},config=run_config)
	if hparams.predict_only:
		predict_input = lambda: get_dataset(FP_train, hparams.train_batch_size, epochs=1)
		for result in estimator.predict(input_fn=predict_input):
			save_numpy_gcs('%s/fov_%d.%d-%d.npy' % (hparams.predict_dir,result['fov'], result['offset_r'], result['offset_col']), result['decompressed_data'])
	else:
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
			

def train_and_evaluate_sequential(hparams):
	# Load composition matrix
	Phi = ccae.load_numpy_gcs('%s/phi.npy' % hparams.train_files).astype('float32')
	Phi = (Phi.T/Phi.sum(1)).T
	U = ccae.load_numpy_gcs('%s/gene_modules.npy' % hparams.train_files).astype('float32')
	PriorAbundance = ccae.load_numpy_gcs('%s/relative_abundance.npy' % hparams.train_files).astype(np.float32)
	Corr = ccae.load_numpy_gcs('%s/correlations.npy' % hparams.train_files).astype('float32')
	# Train the AutoEncoder
	fov_range = [int(fov) for fov in hparams.fovs.split('-')]
	fov_start = fov_range[0]
	FP_train = ['%s/fov_%s.tfrecords' % (hparams.train_files,fov) for fov in range(fov_range[0],fov_range[1])]
	train_input = lambda: ccae.get_dataset(FP_train, hparams.train_batch_size, epochs=hparams.num_epochs_ae)
	train_spec = tf.estimator.TrainSpec(train_input, max_steps=hparams.train_steps)
	eval_input = lambda: ccae.get_dataset(FP_train, hparams.eval_batch_size)
	eval_spec = tf.estimator.EvalSpec(eval_input, steps=hparams.eval_steps)
	# To use multi-gpu need to wait for google folk to install nccl on training instances
	# see: https://stackoverflow.com/questions/52954413/gcmle-notfounderror-libnccl-so-2-cannot-open-shared-object-file
	# Once this works, might also want to prefetch, like here: https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
	if hparams.mirrored_strategy:
		strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
		run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var(), train_distribute=strategy)
	else:
		run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var(),save_checkpoints_secs=6000)
	run_config = run_config.replace(model_dir=hparams.job_dir)
	print('Model dir %s' % run_config.model_dir)
	if not hparams.predict_only:
		estimator = tf.estimator.Estimator(model_fn=ccae.build_estimator_ae,params={'hparams': hparams, 'num_fov': len(FP_train), 'fov_start': fov_start, 'Phi': Phi, 'U': U, 'PriorAbundance': PriorAbundance, 'gene_correlation': Corr},config=run_config)
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	# Train the Decompression
	train_input = lambda: ccae.get_dataset(FP_train, hparams.train_batch_size, epochs=hparams.num_epochs_decompression)
	train_spec = tf.estimator.TrainSpec(train_input, max_steps=hparams.decompress_steps)
	eval_input = lambda: ccae.get_dataset(FP_train, hparams.eval_batch_size)
	eval_spec = tf.estimator.EvalSpec(eval_input, steps=hparams.eval_steps)
	estimator = tf.estimator.Estimator(model_fn=ccae.build_estimator_decompress,params={'hparams': hparams, 'num_fov': len(FP_train), 'fov_start': fov_start, 'Phi': Phi, 'U': U, 'PriorAbundance': PriorAbundance, 'gene_correlation': Corr}, config=run_config)
	if hparams.predict_only:
		print(hparams.fovs)
		predict_input = lambda: ccae.get_dataset(FP_train, hparams.train_batch_size, epochs=1)
		for result in estimator.predict(input_fn=predict_input):
			ccae.save_numpy_gcs('%s/fov_%d.%d-%d.npy' % (hparams.predict_dir,result['fov'], result['offset_r'], result['offset_col']), result['decompressed_data'])
			ccae.save_numpy_gcs('%s_encoded/fov_%d.%d-%d.npy' % (hparams.predict_dir,result['fov'], result['offset_r'], result['offset_col']), result['encoded_data'])
	else:
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Input Arguments
	parser.add_argument('--predict-only', help='Run predictions from previously-trained model',dest='predict_only', action='store_true')
	parser.add_argument('--mirrored-strategy', help='Use a mirrored strategy to distribute across gpus',dest='mirrored_strategy', action='store_true')
	parser.add_argument('--model-type', help='Combined or sequential training', choices=['sequential','combined'], default='sequential')
	parser.add_argument('--train-files', help='GCS file or local paths to training data')
	parser.add_argument('--eval-files', help='GCS file or local paths to evaluation data')
	parser.add_argument('--train-steps',help="""Steps to run the training job for. If --num-epochs is not specified, this must be. Otherwise the training job will run indefinitely.""", default=100, type=int)
	parser.add_argument('--decompress-steps', help='Number of steps to run decompression for at each checkpoint', default=1000, type=int)
	parser.add_argument('--eval-steps', help='Number of steps to run evalution for at each checkpoint', default=1, type=int)
	parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
	parser.add_argument('--predict-dir', help='Where to write the output of predictions')
	parser.add_argument('--fovs', help='Range of fields-of-view to analyze (eg 1-12). Must be in contiguous, increasing order.')
	parser.add_argument('--image-dim', help='Height and width of each image patch',type=int, default=256)
	parser.add_argument('--full-dim', help='Height and width of entire image',type=int, default=1024)
	parser.add_argument('--num-epochs-ae', help='Number of epochs during AE training',type=int, default=100)
	parser.add_argument('--num-epochs-decompression', help='Number of epochs during decompression',type=int, default=5000)
	parser.add_argument('--train-batch-size', help='Batch size for training steps', type=int, default=2)
	parser.add_argument('--eval-batch-size', help='Batch size for evaluation steps', type=int, default=48)
	parser.add_argument('--loss-fn', help='Loss function applied to determine image similarity', choices=['ms-ssim','l1','mse'], default='mse')
	parser.add_argument('--num-filters', help='Number of filters at each layer in the CNN', default=10, type=int)
	parser.add_argument('--num-layers', help='Number of encoding layers in the CNN', default=4, type=int)
	parser.add_argument('--sparsity-k', help='Target sparsity of weights in encoding', default=2.0, type=float)
	parser.add_argument('--lambdaW', help='Lambda for k-sparsity penalty', default=4e-3, type=float)
	parser.add_argument('--lambda-decode', help='Lambda for sparsity of decoding', default=1e-5, type=float)
	parser.add_argument('--lambda-tv', help='Lambda for total variation penalty', default=5e-7, type=float)
	parser.add_argument('--abundance-factor', help='Factor to divide sparsity prior when applied to individual genes', default=9, type=float)
	parser.add_argument('--lambda-abundance-factor', help='Factor to multiply lambda-decode when applied to individual genes', default=3, type=float)
	parser.add_argument('--pixel-sparsity', help='Target pixel sparsity of decoded images', default=0.05, type=float)
	parser.add_argument('--lambda-gene-correlation', help='Factor to multiply correlation penalty', default=1, type=float)
	parser.add_argument('--export-format', help='The input format of the exported SavedModel binary', choices=['JSON', 'CSV', 'EXAMPLE'], default='JSON')
	parser.add_argument('--verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'], default='INFO')
	parser.set_defaults(predict_only=False)
	parser.set_defaults(mirrored_strategy=False)
	args, _ = parser.parse_known_args()
	# Set python level verbosity
	tf.logging.set_verbosity(args.verbosity)
	# Set C++ Graph Execution level verbosity
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[args.verbosity] / 10)
	# Run the training job
	hparams = hparam.HParams(**args.__dict__)
	if hparams.model_type == 'sequential':
		train_and_evaluate_sequential(hparams)
	elif hparams.model_type == 'combined':
		train_and_evaluate_combined(hparams)

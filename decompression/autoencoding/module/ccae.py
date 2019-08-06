import numpy as np
import tensorflow as tf
import glob,os,sys
from scipy.spatial import distance
from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.core.framework.summary_pb2 import Summary
import io
from tensorflow.python.lib.io import file_io

from scipy.stats import entropy
import sys

def _parse_function(example):
	features = {"raw_data": tf.FixedLenFeature([], tf.string),
	"height": tf.FixedLenFeature((), tf.int64),
	"width": tf.FixedLenFeature((), tf.int64),
	"channels": tf.FixedLenFeature((), tf.int64),
	"tissue": tf.FixedLenFeature([], tf.string),
	"fov": tf.FixedLenFeature((), tf.int64),
	"row_offset": tf.FixedLenFeature((), tf.int64),
	"col_offset": tf.FixedLenFeature((), tf.int64),
	"raw_validation": tf.FixedLenFeature([], tf.string),
	"validation_indices": tf.FixedLenSequenceFeature((), tf.int64,True),
	"n_cells": tf.FixedLenFeature((), tf.int64),
	"max_pixels_per_cell": tf.FixedLenFeature((), tf.int64),
	"sp_embedding_mask_row": tf.FixedLenSequenceFeature((), tf.int64,True),
	"sp_embedding_mask_col": tf.FixedLenSequenceFeature((), tf.int64,True),
	"sp_embedding_mask_values": tf.FixedLenSequenceFeature((), tf.int64,True)}
	parsed_features = tf.parse_single_example(example, features)
	raw_data = tf.decode_raw(parsed_features['raw_data'],tf.float32)
	raw_data = raw_data - tf.reduce_min(raw_data)
	raw_validation = tf.decode_raw(parsed_features['raw_validation'],tf.float32)
	raw_validation = raw_validation - tf.reduce_min(raw_validation)
	indices = tf.stack([parsed_features['sp_embedding_mask_row'], parsed_features['sp_embedding_mask_col']],axis=1)
	values = parsed_features['sp_embedding_mask_values']
	# need all to be the same shape, so just using some big number as proxy for max_pixels_per_cell
	shape = [parsed_features['n_cells'], 15000]
	sparse_embedding_tensor = tf.sparse.SparseTensor(indices, values, shape)
	# tf's padded batch breaks with sparse tensors...
	sparse_embedding_tensor = tf.sparse.to_dense(sparse_embedding_tensor)
	return {'features': (raw_data,
						parsed_features['height'],
						parsed_features['width'],
						parsed_features['channels'],
						parsed_features['tissue'],
						parsed_features['fov'],
						parsed_features['row_offset'],
						parsed_features['col_offset'],
						sparse_embedding_tensor),
	'labels': (raw_validation, parsed_features['validation_indices'])}

def _parse_function_withAugment(example):
	features = {"raw_data": tf.FixedLenFeature([], tf.string), "height": tf.FixedLenFeature((), tf.int64), "width": tf.FixedLenFeature((), tf.int64), "channels": tf.FixedLenFeature((), tf.int64), "tissue": tf.FixedLenFeature([], tf.string), "fov": tf.FixedLenFeature((), tf.string), "row_offset": tf.FixedLenFeature((), tf.int64), "col_offset": tf.FixedLenFeature((), tf.int64)}
	parsed_features = tf.parse_single_example(example, features)
	raw_data = tf.decode_raw(parsed_features['raw_data'],tf.float32)
	raw_data = raw_data - tf.reduce_min(raw_data)
	processed_data = augment_data(raw_data)
	return processed_data, parsed_features['height'], parsed_features['width'], parsed_features['channels'], parsed_features['tissue'], parsed_features['fov'], parsed_features['row_offset'], parsed_features['col_offset']

def get_dataset(filepath_list,batch_size,epochs=1,shuffle=True,batch_repeats=0):
	dataset = tf.data.TFRecordDataset(filepath_list)
	if batch_repeats > 0:
		dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(batch_repeats))
	dataset = dataset.map(_parse_function)
	if shuffle:
		dataset = dataset.shuffle(buffer_size=50)
	dataset = dataset.repeat(epochs)
	dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={'features': ([None], [] ,[], [], [], [], [], [], [None,15000]), 'labels': ([None], [None])})
	dataset = dataset.prefetch(buffer_size=None)
	return dataset

def augment_data(input_data, angle=5, shift=5):
	num_images_ = tf.shape(input_data)[0]
	# random rotate
	processed_data = tf.contrib.image.rotate(input_data, tf.random_uniform([num_images_], maxval=np.pi / 180 * angle, minval=np.pi / 180 * -angle))
	# random shift
	base_row = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], shape=[1, 8], dtype=tf.float32)
	base_ = tf.tile(base_row, [num_images_, 1])
	mask_row = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], shape=[1, 8], dtype=tf.float32)
	mask_ = tf.tile(mask_row, [num_images_, 1])
	random_shift_ = tf.random_uniform([num_images_, 8], minval=-shift, maxval=shift, dtype=tf.float32)
	transforms_ = base_ + random_shift_ * mask_
	processed_data = tf.contrib.image.transform(images=processed_data, transforms=transforms_)
	return processed_data

def remove_channel(sample,to_remove):
	ind = tf.concat([tf.range(0, to_remove), tf.range(to_remove + 1, tf.shape(sample)[3])],0)
	removed = tf.gather(sample, ind, axis=3)
	return removed

def _weight_variable(shape, name='weights', regularization=False, clip_value=(),clip_norm=(),init_type=None,init_var=0.1):
	if init_type == 'truncated_normal':
		initial = tf.truncated_normal_initializer(0, 0.1)
	else:
		initial = None
	if (len(clip_value) > 0) and (len(clip_norm) == 0):
		var = tf.get_variable(name, shape, tf.float32, initializer=initial, constraint=lambda x: tf.clip_by_value(x, clip_value[0], clip_value[1]))
	else:
		var = tf.get_variable(name, shape, tf.float32,initializer=initial)
	return var

def shared_conv2d(input,filter_height, filter_width,num_filters,strides=[1,2,2,1],padding='SAME'):
	# batch_size x in_w x in_h x channels x num_filters
	if len(input.get_shape()) == 4:
		input = tf.expand_dims(input,axis=-1)
	# apply one set of filters to all channels, but don't convolve across channels
	filter = _weight_variable([filter_height,filter_width,input.get_shape()[-1],num_filters])
	x = tf.map_fn(lambda xc: tf.nn.conv2d(xc,filter,strides,padding), tf.transpose(input,perm=[3,0,1,2,4]))
	# batch_size x in_w x in_h x channels x num_filters
	x = tf.transpose(x,perm=[1,2,3,0,4])
	return x

def shared_conv2d_transpose(input,filter_height, filter_width,num_filters,strides=[1,2,2,1],padding='SAME'):
	in_shape = input.get_shape()
	filter = _weight_variable([filter_height, filter_width, num_filters, in_shape[-1]])
	out_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1]*strides[1], tf.shape(input)[2]*strides[2], num_filters])
	x = tf.map_fn(lambda xc: tf.nn.conv2d_transpose(xc,filter,out_shape,strides,padding=padding), tf.transpose(input,perm=[3,0,1,2,4]))
	x = tf.transpose(x,perm=[1,2,3,0,4])
	return x

def _encode(x,filter_sizes,stride_factor):
	strides = [1,stride_factor,stride_factor,1]
	for i,(filter_height,filter_width,num_filters) in enumerate(filter_sizes):
		with tf.variable_scope('Layer{}'.format(i+1),reuse=tf.AUTO_REUSE):
			x = shared_conv2d(x,filter_height,filter_width,num_filters,strides=strides)
			x = tf.nn.relu(x)
	return x

def _decode(x,filter_sizes,stride_factor):
	strides = [1,stride_factor,stride_factor,1]
	for i,(filter_height,filter_width,num_filters) in enumerate(filter_sizes):
		with tf.variable_scope('Layer{}'.format(i+1),reuse=tf.AUTO_REUSE):
			x = shared_conv2d_transpose(x,filter_height,filter_width,num_filters,strides=strides)
			if i == len(filter_sizes)-1:
				x = x[:,:,:,:,0]
				x = tf.nn.relu(x)
			else:
				x = tf.nn.relu(x)
	return x	

def encode_and_decode(encode_filters,decode_filters,stride_factor):
	def encode(input,augment,to_remove=None):
		shape = tf.stack([[-1],input[1][:1], input[2][:1], input[3][:1]],0)[:,0]
		data = tf.reshape(input[0],shape)
		if to_remove is not None:
			data = remove_channel(data,to_remove)
		data = tf.cond(augment, lambda: augment_data(data), lambda: data)
		with tf.variable_scope('Encode', reuse=tf.AUTO_REUSE):
			encode_sample = _encode(data,encode_filters,stride_factor)
		sample = [data] + list(input[4:])
		return sample,encode_sample
	def decode(input):
		with tf.variable_scope('Decode', reuse=tf.AUTO_REUSE):
			decode_sample = _decode(input,decode_filters,stride_factor)
		return decode_sample
	return encode,decode

def composite_latent_encoding(x,Phi,U,image_size,patch_size,fov_start,fovs,offset_row,offset_col):
	# input: batch_size x height x width x old_channels x num_filters
	# output: batch_size x height x width x new_channels x num_filters
	# fovs should be contiguous..will offset so min(fov) is index 0
	in_shape = x.get_shape()
	mod_shape = [-1, patch_size[0], patch_size[1], in_shape[4]]
	with tf.variable_scope('Decompress', reuse=tf.AUTO_REUSE):
		W = _weight_variable([U.shape[1]] + [image_size[0], image_size[1], image_size[2], in_shape[4]],init_type='truncated_normal')
	W_idx = tf.stack([fovs-fov_start,offset_row,offset_col], axis=1)
	W_reshape = tf.map_fn(lambda i: W[:,i[0],i[1]:i[1]+patch_size[0],i[2]:i[2]+patch_size[1],:], W_idx, dtype=tf.float32)
	# dict_size x (batch_size x height x width x num_filters)
	W_reshape = tf.reshape(tf.transpose(W_reshape,perm=[1,0,2,3,4]),[U.shape[1],-1])
	# new_channels x (batch_size x height x width x num_filters)
	x_hat = tf.matmul(U,W_reshape)
	x_hat = tf.nn.relu(x_hat)
	# old_channels x (batch_size x height x width x num_filters)
	y = tf.matmul(Phi,x_hat)
	x_hat = tf.reshape(x_hat,[x_hat.get_shape()[0]] + mod_shape)
	x_hat = tf.transpose(x_hat,perm=[1,2,3,0,4])
	y = tf.reshape(y,[y.get_shape()[0]] + mod_shape)
	y = tf.transpose(y,perm=[1,2,3,0,4])
	return y,x_hat,W

def latent_gamma(shape,concentration,rate):
	W = ed.Gamma(concentration=concentration, rate=rate, sample_shape=shape)
	return W

def latent_poisson(shape,rate):
	rate = tf.constant(rate,shape=shape)
	prior = ed.Poisson(rate=rate)
	return prior

def latent_normal(shape,mean,stdev):
	if not isinstance(mean,list):
		mean = tf.constant(mean,shape=shape)
		stdev = tf.constant(stdev,shape=shape)
	prior = ed.Normal(loc=mean,scale=stdev)
	return prior

def get_entropy(W,collapse_channels=False):
	W_ent = tf.abs(W)
	if collapse_channels:
		W_ent = tf.reduce_sum(W_ent,axis=-1)
	W_ent = W_ent/(tf.reduce_sum(W_ent,axis=0) + 1e-5)
	W_ent = tf.log(W_ent + 1e-5)*W_ent
	W_ent = tf.exp(-tf.reduce_sum(W_ent,axis=0))
	# having issues with inf values...
	W_ent = tf.clip_by_value(W_ent,0,tf.cast(tf.shape(W)[0],tf.float32))
	return W_ent

def composite_decoding(x,Phi,image_dim):
	in_shape = x.get_shape()
	y = tf.reshape(x,[-1, in_shape[-1]])
	y = tf.matmul(y,tf.transpose(Phi))
	y = tf.reshape(y,[tf.shape(x)[0], image_dim, image_dim, y.get_shape()[-1]])
	return y

def get_total_variation(x):
	tv = tf.map_fn(lambda xi: tf.image.total_variation(tf.expand_dims(xi,-1)), tf.transpose(x,perm=[3,0,1,2]))
	tv = tf.transpose(tv)
	return tv

def get_eval_summary(EvalImages,EvalImageIndices,DecompressedImages,image_dim):
	# This assumes every example in the batch has the same set of validation genes
	# (which will generally be true, as long as all examples come from the same tissue)
	EvalImages = tf.reshape(EvalImages,[-1,image_dim,image_dim,tf.shape(EvalImageIndices)[1]])
	#Decompressed_subset = tf.gather(DecompressedImages,EvalImageIndices[0],axis=-1)
	Decompressed_subset = tf.map_fn(lambda de: tf.gather(de[0],de[1],axis=-1), (DecompressedImages,EvalImageIndices),dtype=tf.float32)
	return EvalImages,Decompressed_subset

def load_numpy_gcs(path,mode='rb'):
	try:
		x = np.load(path)
	except:
		f_stream = file_io.FileIO(path, mode)
		x = np.load(io.BytesIO(f_stream.read()) )
	return x

def save_numpy_gcs(path,object):
	try:
		np.save(path,object)
	except:
		np.save(file_io.FileIO(path, 'w'), object)

def ssim_metric(im1,im2):
	return tf.metrics.mean(tf.image.ssim_multiscale(im1,tf.div(im2,tf.reduce_max(im2)),1))

def l1_metric(im1,im2):
	return tf.metrics.mean(tf.losses.absolute_difference(tf.math.log(im1+1e-5),tf.math.log(im2+1e-5)))

def mse_combined_metric(tensor_tuples):
	return tf.metrics.mean([tf.reduce_mean(tf.square(t1-t2)) for t1,t2 in tensor_tuples])

def resize_correlation_metric(im1,im2,size,factor=4):
	new_size = tf.constant([int(size/factor),int(size/factor)])
	im1_resize = tf.image.resize_images(im1,new_size)
	im2_resize = tf.image.resize_images(im2,new_size)
	m1,s1 = tf.nn.moments(im1_resize,axes=[0,1,2])
	m2,s2 = tf.nn.moments(im2_resize,axes=[0,1,2])
	m12,s12 = tf.nn.moments(tf.multiply(im1_resize-m1,im2_resize-m2),axes=[0,1,2])
	corr = tf.div(m12, tf.multiply(tf.sqrt(s1), tf.sqrt(s2)) + 1e-5)
	return tf.metrics.mean(corr)

def correlation_metric(im1,im2):
	# inputs: cells x channels
	m1,s1 = tf.nn.moments(im1,axes=[0])
	m2,s2 = tf.nn.moments(im2,axes=[0])
	m12,s12 = tf.nn.moments(tf.multiply(im1-m1,im2-m2),axes=[0])
	corr = tf.div(m12, tf.multiply(tf.sqrt(s1), tf.sqrt(s2)) + 1e-5)
	return tf.metrics.mean(corr)

def correlation_matrix(x, reshape_first=False,rescale=False):
	if reshape_first:
		x = tf.reduce_sum(x,axis=4)
		if rescale:
			x = tf.image.resize_images(x,[tf.shape(x)[1]/4,tf.shape(x)[2]/4])
		x = tf.transpose(x,[3,0,1,2])
		x = tf.reshape(x,[tf.shape(x)[0],-1])
	m,s = tf.nn.moments(x,axes=[1])
	mx = tf.matmul(m[...,None],m[None,...])
	vx = tf.matmul(x,tf.transpose(x))/tf.cast(tf.shape(x[1]),tf.float32)
	sx = tf.matmul(tf.sqrt(s)[...,None],tf.sqrt(s)[None,...])
	corr = tf.div(vx - mx,sx + 1e-5)
	ones = tf.ones_like(corr)
	mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
	mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
	mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask
	return tf.boolean_mask(corr,mask)

def sparse_embedding_single_example(images, dense_embedding_indices):
	# images input shape: height x width x channels
	im_shape = tf.shape(images)
	im_reshape = tf.reshape(images,[-1,im_shape[2]])
	sparse_embedding_indices = tf.contrib.layers.dense_to_sparse(dense_embedding_indices)
	x = tf.nn.embedding_lookup_sparse(im_reshape,sparse_embedding_indices,None,combiner='sum')
	zero_pad_size = tf.shape(dense_embedding_indices)[0] - tf.shape(x)[0]
	x = tf.concat([x,tf.zeros([zero_pad_size, tf.shape(x)[1]])], axis=0)
	return x

def batch_sparse_embedding(batch_images, batch_dense_embedding_indices):
	cell_integrated_intensities = tf.map_fn(lambda x: sparse_embedding_single_example(x[0],x[1]), (batch_images, batch_dense_embedding_indices), dtype=tf.float32)
	cell_integrated_intensities = tf.reshape(cell_integrated_intensities,[-1,tf.shape(cell_integrated_intensities)[2]])
	# remove padded empty cells
	nnz = tf.count_nonzero(cell_integrated_intensities,axis=1)
	idx = tf.where(tf.not_equal(nnz,0))
	cell_integrated_intensities = tf.gather_nd(cell_integrated_intensities,idx)
	# output shape is cells x channels
	return cell_integrated_intensities	

def build_estimator_ae(features, mode, params):
	next_element = features['features']
	validation_data,validation_indices = features['labels']
	hparams = params['hparams']
	# Load composition matrix
	Phi = params['Phi']
	U = params['U']
	PriorAbundance = params['PriorAbundance']
	# Training Params
	lr_auto = 1e-2
	# Network Inputs
	lr_autoencode = lr_auto#tf.placeholder(tf.float32,())
	ph_dropout = 1#tf.placeholder(tf.float32, (), 'dropout')
	phase = 0#tf.placeholder(tf.bool, name='phase')
	augment = tf.placeholder_with_default(False, shape=())
	# Network Architecture
	encode_filters = [[3,3,hparams.num_filters] for _ in range(hparams.num_layers)]
	decode_filters = [[3,3,hparams.num_filters]]*(hparams.num_layers-1) +  [[3,3,1]]
	stride_factor = 2
	# Target sparsity of decoding
	sparsity_decode = hparams.pixel_sparsity*hparams.image_dim**2
	# Build AutoEncoder
	encode,decode = encode_and_decode(encode_filters,decode_filters,stride_factor)
	sample,encode_sample = encode(next_element,augment)
	decode_sample = decode(encode_sample)
	prior_decode = latent_normal([Phi.shape[0]],sparsity_decode,sparsity_decode)
	decode_entropy = get_entropy(tf.reshape(decode_sample,(-1,tf.shape(decode_sample)[-1])))
	# Build AE loss
	if hparams.loss_fn == 'ms-ssim':
		auto_loss = -tf.reduce_mean(tf.image.ssim_multiscale(sample[0],tf.div(decode_sample,tf.reduce_max(decode_sample)),1))
	elif hparams.loss_fn == 'l1':
		auto_loss = tf.reduce_mean(tf.losses.absolute_difference(tf.math.log(sample[0]+1e-5),tf.math.log(decode_sample+1e-5)))
	elif hparams.loss_fn == 'mse':
		auto_loss = tf.reduce_mean(tf.square(sample[0] - decode_sample))
	auto_loss += -tf.reduce_mean(prior_decode.distribution.log_prob(decode_entropy))*hparams.lambda_decode
	auto_loss += tf.reduce_mean(get_total_variation(decode_sample))*hparams.lambda_tv
	# Build graph for decompression even though we don't update in this estimator
	sample_decompress,encode_sample_decompress = encode(next_element,augment)
	decode_sample_decompress = decode(encode_sample_decompress)
	encode_size = int(hparams.full_dim/(stride_factor**(len(encode_filters))))
	encode_patch_size = int(hparams.image_dim/(stride_factor**(len(encode_filters))))
	offset_row = tf.cast(sample_decompress[3]/(stride_factor**(len(encode_filters))),tf.int64)
	offset_col = tf.cast(sample_decompress[4]/(stride_factor**(len(encode_filters))),tf.int64)
	encode_sample_fit, encode_latent, W = composite_latent_encoding(encode_sample_decompress, Phi, U, (params['num_fov'],encode_size,encode_size), (encode_patch_size, encode_patch_size), params['fov_start'], sample_decompress[2], offset_row, offset_col)
	prior_W = latent_poisson(W.get_shape()[1:-1],hparams.sparsity_k)
	W_entropy = get_entropy(W,collapse_channels=True)
	decode_latent = decode(encode_latent)
	compose_latent = composite_decoding(decode_latent,Phi,hparams.image_dim)
	cell_intensities = batch_sparse_embedding(decode_latent, next_element[8])
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions={'tissue': sample_decompress[1], 'fov': sample_decompress[2], 'offset_r': sample_decompress[3], 'offset_col': sample_decompress[4],'decompressed_data': decode_latent, 'encoded_data': encode_latent})
	prior_decode_latent = latent_normal(None,PriorAbundance*sparsity_decode/hparams.abundance_factor, sparsity_decode/hparams.abundance_factor)
	decode_latent_entropy = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: get_entropy(cell_intensities), false_fn= lambda: tf.zeros([tf.shape(cell_intensities)[1]]))
	gene_correlation = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: correlation_matrix(encode_latent,True), false_fn= lambda: tf.zeros([len(params['gene_correlation'])]))
	#gene_correlation = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: correlation_matrix(tf.transpose(cell_intensities)), false_fn= lambda: tf.zeros([len(params['gene_correlation'])]))
	encode_loss = tf.reduce_mean(tf.square(encode_sample_decompress - encode_sample_fit))
	if hparams.loss_fn == 'ms-ssim':
		decode_loss = -tf.reduce_mean(tf.image.ssim_multiscale(sample_decompress[0],tf.div(compose_latent,tf.reduce_max(compose_latent)),1))
	elif hparams.loss_fn == 'l1':
		decode_loss = tf.reduce_mean(tf.losses.absolute_difference(tf.math.log(decode_sample_decompress+1e-5),tf.math.log(compose_latent+1e-5)))
	elif hparams.loss_fn == 'mse':
		decode_loss = tf.reduce_mean(tf.square(decode_sample_decompress - compose_latent))
	# Decompression loss
	reg_loss = -tf.reduce_mean(prior_W.distribution.log_prob(W_entropy))*hparams.lambdaW
	reg_loss += -tf.reduce_mean(prior_decode_latent.distribution.log_prob(decode_latent_entropy))*hparams.lambda_decode*hparams.lambda_abundance_factor
	reg_loss += tf.reduce_mean(get_total_variation(decode_latent))*hparams.lambda_tv
	reg_loss += tf.reduce_mean(tf.square(gene_correlation - params['gene_correlation']))*hparams.lambda_gene_correlation
	decompress_loss = encode_loss + decode_loss + reg_loss
	summary_im1,summary_im2 = get_eval_summary(validation_data,validation_indices,decode_latent,hparams.image_dim)
	cell_intensities_im1 = batch_sparse_embedding(summary_im1, next_element[8])
	cell_intensities_im2 = batch_sparse_embedding(summary_im2, next_element[8])
	# Build optimizers
	optimizer_decomp = tf.train.AdamOptimizer(name='Adam_decomp',learning_rate=lr_autoencode, beta1=0.9, beta2=0.999)
	decompress_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decompress')
	# Create training operations
	train_decompress = optimizer_decomp.minimize(decompress_loss, var_list=decompress_vars, global_step=tf.train.get_global_step())
	if mode == tf.estimator.ModeKeys.EVAL:
		if hparams.loss_fn == 'ms-ssim':
			metrics = {'ms-ssim': ssim_metric(sample[0], decode_sample)}
		elif hparams.loss_fn == 'l1':
			metrics = {'l1_ae': l1_metric(sample[0], decode_sample)}
		elif hparams.loss_fn == 'mse':
			metrics = {'mse': tf.metrics.mean_squared_error(sample[0],decode_sample)}
		return tf.estimator.EstimatorSpec(mode, loss=auto_loss, eval_metric_ops=metrics)
	if mode == tf.estimator.ModeKeys.TRAIN:
		# Build optimizers
		optimizer_auto = tf.train.AdamOptimizer(name='Adam_ae',learning_rate=lr_autoencode, beta1=0.9, beta2=0.999)
		encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encode')
		decode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decode')
		# Create training operations (for batch_norm)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_auto = optimizer_auto.minimize(auto_loss, var_list=encode_vars+decode_vars, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=auto_loss, train_op=train_auto)

def build_estimator_decompress(features, mode, params):
	next_element = features['features']
	validation_data,validation_indices = features['labels']
	hparams = params['hparams']
	# Load gene modules and composition matrix
	Phi = params['Phi']
	U = params['U']
	PriorAbundance = params['PriorAbundance']
	# Training Params
	lr_auto = 2e-2
	# Network Inputs
	lr_autoencode = lr_auto#tf.placeholder(tf.float32,())
	ph_dropout = 1#tf.placeholder(tf.float32, (), 'dropout')
	phase = 0#tf.placeholder(tf.bool, name='phase')
	augment = tf.placeholder_with_default(False, shape=())
	# Network Architecture
	encode_filters = [[3,3,hparams.num_filters] for _ in range(hparams.num_layers)]
	decode_filters = [[3,3,hparams.num_filters]]*(hparams.num_layers-1) +  [[3,3,1]]
	stride_factor = 2
	# Target sparsity of decoding
	sparsity_decode = hparams.pixel_sparsity*hparams.image_dim**2
	# Build AutoEncoder
	encode,decode = encode_and_decode(encode_filters,decode_filters,stride_factor)
	# Build decompression
	sample_decompress,encode_sample_decompress = encode(next_element,augment)
	decode_sample_decompress = decode(encode_sample_decompress)
	encode_size = int(hparams.full_dim/(stride_factor**(len(encode_filters))))
	encode_patch_size = int(hparams.image_dim/(stride_factor**(len(encode_filters))))
	offset_row = tf.cast(sample_decompress[3]/(stride_factor**(len(encode_filters))),tf.int64)
	offset_col = tf.cast(sample_decompress[4]/(stride_factor**(len(encode_filters))),tf.int64)
	encode_sample_fit, encode_latent, W = composite_latent_encoding(encode_sample_decompress, Phi, U, (params['num_fov'],encode_size,encode_size), (encode_patch_size, encode_patch_size), params['fov_start'], sample_decompress[2], offset_row, offset_col)
	prior_W = latent_poisson(W.get_shape()[1:-1],hparams.sparsity_k)
	W_entropy = get_entropy(W,collapse_channels=True)
	decode_latent = decode(encode_latent)
	compose_latent = composite_decoding(decode_latent,Phi,hparams.image_dim)
	cell_intensities = batch_sparse_embedding(decode_latent, next_element[8])
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions={'tissue': sample_decompress[1], 'fov': sample_decompress[2], 'offset_r': sample_decompress[3], 'offset_col': sample_decompress[4],'decompressed_data': decode_latent, 'encoded_data': encode_latent})
	prior_decode_latent = latent_normal(None,PriorAbundance*sparsity_decode/hparams.abundance_factor, sparsity_decode/hparams.abundance_factor)
	decode_latent_entropy = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: get_entropy(cell_intensities), false_fn= lambda: tf.zeros([tf.shape(cell_intensities)[1]]))
	gene_correlation = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: correlation_matrix(encode_latent,True), false_fn= lambda: tf.zeros([len(params['gene_correlation'])]))
	#gene_correlation = tf.cond(tf.reduce_sum(cell_intensities) > 0, true_fn= lambda: correlation_matrix(tf.transpose(cell_intensities)), false_fn= lambda: tf.zeros([len(params['gene_correlation'])]))
	encode_loss = tf.reduce_mean(tf.square(encode_sample_decompress - encode_sample_fit))
	if hparams.loss_fn == 'ms-ssim':
		decode_loss = -tf.reduce_mean(tf.image.ssim_multiscale(sample_decompress[0],tf.div(compose_latent,tf.reduce_max(compose_latent)),1))
	elif hparams.loss_fn == 'l1':
		decode_loss = tf.reduce_mean(tf.losses.absolute_difference(tf.math.log(decode_sample_decompress+1e-5),tf.math.log(compose_latent+1e-5)))
	elif hparams.loss_fn == 'mse':
		decode_loss = tf.reduce_mean(tf.square(decode_sample_decompress - compose_latent))
	# Decompression loss
	reg_loss = -tf.reduce_mean(prior_W.distribution.log_prob(W_entropy))*hparams.lambdaW
	reg_loss += -tf.reduce_mean(prior_decode_latent.distribution.log_prob(decode_latent_entropy))*hparams.lambda_decode*hparams.lambda_abundance_factor
	reg_loss += tf.reduce_mean(get_total_variation(decode_latent))*hparams.lambda_tv
	reg_loss += tf.reduce_mean(tf.square(gene_correlation - params['gene_correlation']))*hparams.lambda_gene_correlation
	decompress_loss = encode_loss + decode_loss + reg_loss
	if mode == tf.estimator.ModeKeys.TRAIN:
		# Build optimizers
		optimizer_decomp = tf.train.AdamOptimizer(name='Adam_decomp',learning_rate=lr_autoencode, beta1=0.9, beta2=0.999)
		decompress_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decompress')
		# Create training operations
		train_decompress = optimizer_decomp.minimize(decompress_loss, var_list=decompress_vars, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=decompress_loss, train_op=train_decompress)
	summary_im1,summary_im2 = get_eval_summary(validation_data,validation_indices,decode_latent,hparams.image_dim)
	cell_intensities_im1 = batch_sparse_embedding(summary_im1, next_element[8])
	cell_intensities_im2 = batch_sparse_embedding(summary_im2, next_element[8])
	metrics = {}
	metrics['mse_encode'] = tf.metrics.mean_squared_error(encode_sample_decompress, encode_sample_fit)
	metrics['resize_corr'] = resize_correlation_metric(summary_im1,summary_im2,hparams.image_dim)
	metrics['segmented_corr'] = correlation_metric(summary_im1,summary_im2)
	metrics['W_entropy'] = tf.metrics.mean(W_entropy)
	if hparams.loss_fn == 'ms-ssim':
		metrics['ssim_validation'] = ssim_metric(summary_im1,summary_im2)
		metrics['ssim_ae'] =  ssim_metric(sample_decompress[0], decode_sample_decompress)
		metrics['ssim_decode'] = ssim_metric(sample_decompress[0], compose_latent)
	elif hparams.loss_fn == 'l1':
		metrics['l1_validation'] = l1_metric(summary_im1,summary_im2)
		metrics['l1_ae'] =  l1_metric(sample_decompress[0], decode_sample_decompress)
		metrics['l1_decode'] = l1_metric(sample_decompress[0], compose_latent)
	elif hparams.loss_fn == 'mse':
		metrics['mse_validation'] = tf.metrics.mean_squared_error(summary_im1,summary_im2)
		metrics['mse_ae'] =  tf.metrics.mean_squared_error(sample_decompress[0], decode_sample_decompress)
		metrics['mse_decode'] = tf.metrics.mean_squared_error(sample_decompress[0], compose_latent)
		metrics['mse_combined'] = mse_combined_metric([(summary_im1,summary_im2),(sample_decompress[0], decode_sample_decompress),(sample_decompress[0], compose_latent)])
	return tf.estimator.EstimatorSpec(mode, loss=decompress_loss, eval_metric_ops=metrics)

def build_estimator_combined(features, mode, params):
	next_element = features['features']
	validation_data,validation_indices = features['labels']
	hparams = params['hparams']
	# Load composition matrix
	Phi = load_numpy_gcs('%s/phi.npy' % hparams.train_files).astype('float32')
	Phi = (Phi.T/Phi.sum(1)).T
	# Training Params
	lr_auto = 1e-3
	# Network Inputs
	lr_autoencode = lr_auto#tf.placeholder(tf.float32,())
	ph_dropout = 1#tf.placeholder(tf.float32, (), 'dropout')
	phase = 0#tf.placeholder(tf.bool, name='phase')
	augment = tf.placeholder_with_default(False, shape=())
	# Network Architecture
	encode_filters = [[3,3,hparams.num_filters] for _ in range(hparams.num_layers)]
	decode_filters = [[3,3,hparams.num_filters]]*(hparams.num_layers-1) +  [[3,3,1]]
	stride_factor = 2
	# Target sparsity of decoding
	sparsity_decode = 0.05*hparams.image_dim**2
	# Build AutoEncoder
	encode,decode = encode_and_decode(encode_filters,decode_filters,stride_factor)
	sample,encode_sample = encode(next_element,augment)
	decode_sample = decode(encode_sample)
	prior_decode = latent_normal([Phi.shape[0]],sparsity_decode,sparsity_decode*1.5)
	decode_entropy = get_entropy(tf.reshape(decode_sample,(-1,tf.shape(decode_sample)[-1])))
	# Build AE loss
	if hparams.loss_fn == 'ms-ssim':
		auto_loss = -tf.reduce_mean(tf.image.ssim_multiscale(sample[0],tf.div(decode_sample,tf.reduce_max(decode_sample)),1))
	elif hparams.loss_fn == 'l1':
		auto_loss = tf.reduce_mean(tf.losses.absolute_difference(tf.math.log(sample[0]+1e-5),tf.math.log(decode_sample+1e-5)))
	elif hparams.loss_fn == 'mse':
		auto_loss = tf.reduce_mean(tf.square(sample[0] - decode_sample))
	auto_loss += -tf.reduce_mean(prior_decode.distribution.log_prob(decode_entropy))*hparams.lambda_decode
	auto_loss += tf.reduce_mean(get_total_variation(decode_sample))*hparams.lambda_tv
	# Build graph for decompression even though we don't update in this estimator
	# Load gene modules and composition matrix
	U = load_numpy_gcs('%s/gene_modules.npy' % hparams.train_files).astype('float32')
	PriorAbundance = load_numpy_gcs('%s/relative_abundance.npy' % hparams.train_files).astype(np.float32)
	sample_decompress,encode_sample_decompress = encode(next_element,augment)
	decode_sample_decompress = decode(encode_sample_decompress)
	encode_size = int(hparams.full_dim/(stride_factor**(len(encode_filters))))
	encode_patch_size = int(hparams.image_dim/(stride_factor**(len(encode_filters))))
	offset_row = tf.cast(sample_decompress[3]/(stride_factor**(len(encode_filters))),tf.int64)
	offset_col = tf.cast(sample_decompress[4]/(stride_factor**(len(encode_filters))),tf.int64)
	encode_sample_fit, encode_latent, W = composite_latent_encoding(encode_sample_decompress, Phi, U, (params['num_fov'],encode_size,encode_size), (encode_patch_size, encode_patch_size), params['fov_start'], sample_decompress[2], offset_row, offset_col)
	prior_W = latent_poisson(W.get_shape()[1:-1],hparams.sparsity_k)
	W_entropy = get_entropy(W,collapse_channels=True)
	decode_latent = decode(encode_latent)
	compose_latent = composite_decoding(decode_latent,Phi,hparams.image_dim)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions={'tissue': sample_decompress[1], 'fov': sample_decompress[2], 'offset_r': sample_decompress[3], 'offset_col': sample_decompress[4],'decompressed_data': decode_latent})
	prior_decode_latent = latent_normal(None,PriorAbundance*sparsity_decode,PriorAbundance*sparsity_decode/hparams.abundance_factor)
	decode_latent_entropy = get_entropy(tf.reshape(decode_latent,(-1,tf.shape(decode_latent)[-1])))
	encode_loss = tf.reduce_mean(tf.square(encode_sample_decompress - encode_sample_fit))
	if hparams.loss_fn == 'ms-ssim':
		decode_loss = -tf.reduce_mean(tf.image.ssim_multiscale(sample_decompress[0],tf.div(compose_latent,tf.reduce_max(compose_latent)),1))
	elif hparams.loss_fn == 'l1':
		decode_loss = tf.reduce_mean(tf.losses.absolute_difference(tf.math.log(sample_decompress[0]+1e-5),tf.math.log(compose_latent+1e-5)))
	elif hparams.loss_fn == 'mse':
		decode_loss = tf.reduce_mean(tf.square(sample_decompress[0] - compose_latent))
	# Decompression loss
	reg_loss = -tf.reduce_mean(prior_W.distribution.log_prob(W_entropy))*hparams.lambdaW
	reg_loss += -tf.reduce_mean(prior_decode_latent.distribution.log_prob(decode_latent_entropy))*hparams.lambda_decode*hparams.lambda_abundance_factor
	reg_loss += tf.reduce_mean(get_total_variation(decode_latent))*hparams.lambda_tv
	decompress_loss = encode_loss + decode_loss + reg_loss
	loss = auto_loss + decompress_loss
	if mode == tf.estimator.ModeKeys.EVAL:
		summary_im1,summary_im2 = get_eval_summary(validation_data,validation_indices,decode_latent,hparams.image_dim)
		metrics = {}
		metrics['mse_encode'] = tf.metrics.mean_squared_error(encode_sample_decompress, encode_sample_fit)
		metrics['resize_corr'] = resize_correlation_metric(summary_im1,summary_im2,hparams.image_dim)
		if hparams.loss_fn == 'ms-ssim':
			metrics['ssim_validation'] = ssim_metric(summary_im1,summary_im2)
			metrics['ssim_ae'] =  ssim_metric(sample_decompress[0], decode_sample_decompress)
			metrics['ssim_decode'] = ssim_metric(sample_decompress[0], compose_latent)
		elif hparams.loss_fn == 'l1':
			metrics['l1_validation'] = l1_metric(summary_im1,summary_im2)
			metrics['l1_ae'] =  l1_metric(sample_decompress[0], decode_sample_decompress)
			metrics['l1_decode'] = l1_metric(sample_decompress[0], compose_latent)
		elif hparams.loss_fn == 'mse':
			metrics['mse_validation'] = tf.metrics.mean_squared_error(summary_im1,summary_im2)
			metrics['mse_ae'] =  tf.metrics.mean_squared_error(sample_decompress[0], decode_sample_decompress)
			metrics['mse_decode'] = tf.metrics.mean_squared_error(sample_decompress[0], compose_latent)
			metrics['mse_combined'] = mse_combined_metric([(summary_im1,summary_im2),(sample_decompress[0], decode_sample_decompress),(sample_decompress[0], compose_latent)])
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
	if mode == tf.estimator.ModeKeys.TRAIN:
		# Build optimizers
		optimizer = tf.train.AdamOptimizer(name='Adam',learning_rate=lr_autoencode, beta1=0.9, beta2=0.999)
		encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encode')
		decode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decode')
		decompress_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decompress')
		# Create training operations (for batch_norm)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss, var_list=encode_vars+decode_vars+decompress_vars, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



# class YClass( object ):
#     pass
# 
# hparams = YClass()
# hp = {'train_files': 'data','num_layers': 5,'image_dim':512, 'full_dim': 2048,'sparsity_k': 2.0, 'lambdaW':4e-3, 'lambda_decode': 1e-5, 'lambda_tv': 5e-7, 'fovs': '0,1,2,3,4', 'train_steps': 100, 'eval_steps': 1, 'train_batch_size': 2, 'eval_batch_size': 2, 'job_dir': 'output', 'num_epochs_ae': 8,'num_epochs_decompression': 750, 'num_filters': 5, 'decompress_steps': 1000}
# for k,v in hp.items():
# 	setattr(hparams,k,v)
# 


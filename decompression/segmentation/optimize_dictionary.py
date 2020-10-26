import numpy as np
import spams
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.sparse import load_npz
import imageio
import tensorflow as tf
import argparse, os

THREADS=40
def sparse_decode(Y,D,lda):
	Ynorm = np.linalg.norm(Y)**2/Y.shape[1]
	W = spams.lasso(np.asfortranarray(Y),np.asfortranarray(D),lambda1=lda*Ynorm,mode=1,numThreads=THREADS,pos=False)
	W = np.asarray(W.todense())
	return W

def sparse_decode_blocks(Y,D,lda,num_blocks=20):
	W = np.zeros((D.shape[1],Y.shape[1]))
	ynorm = np.linalg.norm(Y,axis=0)
	xs = np.argsort(ynorm)
	block_size = int(len(xs)/num_blocks)
	for i in range(0,len(xs),block_size):
		idx = xs[i:i+block_size]
		w = sparse_decode(Y[:,idx],D,lda)
		W[:,idx] = w
	return W

def correlation_matrix(x):
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

def conditional_probability_matrix(x,thresh):
	x_sig = tf.transpose(tf.math.sigmoid(tf.transpose(x)-thresh))
	cp = tf.matmul(x_sig, tf.transpose(x_sig))/(tf.reduce_sum(x_sig,axis=1) + 1e-3)
	return tf.transpose(cp)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--trainpath', help='Path to directory with composition (phi) and gene module (U) matrices')
	parser.add_argument('--modules-in', help='Input name of gene module dictionary', default='gene_modules')
	parser.add_argument('--modules-out', help='Output name of gene module dictionary', default='gene_modules')
	parser.add_argument('--region-mask', help='Mask for cells in a region of interest',default=None)
	parser.add_argument('--method', help='Signal integration method',default='integrated_intensity')
	parser.add_argument('--batch-size', help='Batch size',default=5000,type=int)
	parser.add_argument('--epochs', help='Number of training epochs',default=25,type=int)
	parser.add_argument('--train-fraction', help='Fraction of data to use for training',default=0.6,type=float)
	parser.add_argument('--alpha-cond', help='Regularization parameter for conditional probability',default=1,type=float)
	parser.add_argument('--alpha-corr', help='Regularization parameter for correlation',default=1,type=float)
	args,_ = parser.parse_known_args()
	os.environ['KMP_WARNINGS'] = '0'
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	phi = np.load('%s/phi.npy' % args.trainpath)
	U0 = np.load('%s/%s' % (args.trainpath, args.modules_in)).astype(np.float32)
	f = open('%s/genes.txt' % args.trainpath)
	Genes = np.array([l.strip() for l in f])
	f.close()
	training_abundance = np.load('%s/relative_abundance.npy' % args.trainpath)
	training_correlation = np.load('%s/correlations.npy' % args.trainpath)
	training_conditional_prob = np.load('%s/conditional_probability.npy' % args.trainpath)
	T = [int(t) for t in args.tissues.split(',')]
	Y = []
	X_validation = []
	idx_validation = []
	DirectThresholds = np.zeros(len(Genes))
	for t in T:
		tissue = 'tissue%d' % t
		composite_measurements = np.load('%s/%s/%s/composite_measurements.npy' % (args.basepath,tissue,args.method))
		x_validation = np.load('%s/%s/%s/direct_measurements.npy' % (args.basepath,tissue,args.method))
		direct_labels = np.load('%s/%s/%s/direct_labels.npy' % (args.basepath,tissue,args.method))
		direct_thresholds = np.load('%s/%s/%s/direct_thresholds.npy' % (args.basepath,tissue,args.method))
		x_validation = (x_validation.T - direct_thresholds).T
		x_validation[x_validation < 0] = 0
		CellMasks = load_npz('%s/%s/segmented/cell_masks.size_threshold.npz' % (args.basepath,tissue))
		if args.region_mask is not None:
			region_mask = imageio.imread('%s/%s/stitched_aligned_filtered/%s' % (args.basepath,tissue,args.region_mask))
			cidx = CellMasks.dot(region_mask.flatten())
			cidx = np.where(cidx)[0]
			composite_measurements = composite_measurements[:,cidx]
			x_validation = x_validation[:,cidx]
			CellMasks = CellMasks[cidx]
		idx = [np.where(Genes == l)[0][0] for l in direct_labels]
		Y.append(composite_measurements)
		X_validation.append(x_validation)
		idx_validation.append(np.ones((composite_measurements.shape[1],len(idx)),dtype=np.int)*idx)
		for i,l in enumerate(direct_labels):
			DirectThresholds[np.where(Genes == l)[0][0]] = direct_thresholds[i]
	Y = np.hstack(Y)
	X_validation = np.hstack(X_validation)
	idx_validation = np.vstack(idx_validation).astype(np.int32)
	DirectThresholds[DirectThresholds == 0] = np.average(DirectThresholds)
	gene_scale = []
	for i in range(U0.shape[0]):
		x = X_validation[np.where(idx_validation.T == i)]
		if len(x) > 0:
			gene_scale.append(np.percentile(x[x>1],75)**2)
		else:
			gene_scale.append(1)
	gene_scale = tf.constant(gene_scale)
	W = tf.placeholder(tf.float32,[U0.shape[1],args.batch_size])
	U = tf.Variable(initial_value=U0, constraint=lambda x: tf.clip_by_value(x,0,1))
	xhat = tf.nn.relu(tf.matmul(U,W))
	x0 = tf.placeholder(tf.float32,[len(idx),args.batch_size])
	idx_v = tf.placeholder(tf.int32,[args.batch_size,idx_validation.shape[1]])
	xhat_subset = tf.map_fn(lambda xh: tf.gather(xh[0],xh[1],axis=0), (tf.transpose(xhat),idx_v),dtype=tf.float32)
	scale = tf.gather(gene_scale,tf.reshape(tf.transpose(idx_v),[-1]))
	error = tf.math.square(x0-tf.transpose(xhat_subset))
	# corr_loss and cond_loss scale like squared deviation for small differences, but scale much faster for large differences
	corr_loss = (tf.math.exp(tf.reduce_mean(tf.math.square(training_correlation - correlation_matrix(xhat)))) - 1)
	cond_loss = (tf.math.exp(tf.reduce_mean(tf.math.square(training_conditional_prob - conditional_probability_matrix(xhat,DirectThresholds)))) - 1)
	loss = tf.reduce_mean(tf.reshape(error,[-1])/scale) + cond_loss*args.alpha_cond + corr_loss*args.alpha_corr
	optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
	vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss,var_list=vars)
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	# train on random x% of data
	Uhat = np.copy(U0).astype(np.float32)
	train_idx = np.random.choice(Y.shape[1], int(Y.shape[1]*args.train_fraction), replace=False)
	test_idx = np.setdiff1d(np.arange(Y.shape[1]),train_idx)
	for e in range(args.epochs):
		np.random.shuffle(train_idx)
		for i in range(0,len(train_idx),args.batch_size):
			if len(train_idx[i:i+args.batch_size]) == args.batch_size:
				Uhat = sess.run(U)
				What = sparse_decode_blocks(Y[:,train_idx[i:i+args.batch_size]], phi.dot(Uhat), 0.1)
				if What.sum() > 0: # cause of nan loss?
					_,l = sess.run([train_op,loss], feed_dict={x0: X_validation[:,train_idx[i:i+args.batch_size]], W: What, idx_v: idx_validation[train_idx[i:i+args.batch_size]]})
		w0 = sparse_decode_blocks(Y[:,test_idx],phi.dot(Uhat),0.1)
		xh = Uhat.dot(w0)
		xh[np.isnan(xh)] = 0
		xh_subset = np.array([x[idx] for x,idx in zip(xh.T,idx_validation[test_idx])]).T
		corr_test = 1-distance.correlation(X_validation[:,test_idx].flatten(),xh_subset.flatten())
		ent = np.average([np.exp(entropy(abs(u))) for u in Uhat.T if abs(u).sum()>0])
		print('Epoch: %d; Loss: %f; Entropy: %f; Correlation: %f' % (e+1,l,ent,corr_test))
	np.save('%s/%s.npy' % (args.trainpath, args.modules_out),Uhat)
	np.save('%s/%s.test_idx.npy' % (args.trainpath, args.modules_out),test_idx)
	np.save('%s/%s.train_idx.npy' % (args.trainpath, args.modules_out),train_idx)



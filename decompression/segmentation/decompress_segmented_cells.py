import numpy as np
import argparse, os
from optimize_dictionary import sparse_decode_blocks
import pandas as pd
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.sparse import load_npz
import imageio

def select_and_correct_comeasured(x,xc,phi,phi_corr,training_corr,phi_thresh=0.6,train_thresh=0.1):
	# find comeasured genes that are not coexpressed
	comeasured = []
	for i in range(phi_corr.shape[0]):
		xs = np.argsort(-phi_corr[i])
		for j in xs:
			if phi_corr[i,j] < phi_thresh:
				break
			comeasured.append((phi_corr[i,j],i,j))
	corrected_pairs = []
	for c in sorted(comeasured,reverse=True):
		if training_corr[c[1],c[2]] < train_thresh:
			x, both_nz = correct_coexpression(x,xc,phi,c[1],c[2])
			corrected_pairs.append((c[1],c[2], both_nz))
	return x, corrected_pairs

def correct_coexpression(x,xc,phi,i,j):
	# pick the gene with nearest expression pattern in scRNA
	thresh_i = np.percentile(x[i],99.9)/100
	thresh_j = np.percentile(x[j],99.9)/100
	both_nz = (x[i] > thresh_i)*(x[j] > thresh_j)
	dist = distance.cdist([phi[:,i], phi[:,j]], xc[:,both_nz].T,'correlation')
	i_closer = np.where(both_nz)[0][dist[0] < dist[1]]
	j_closer = np.where(both_nz)[0][dist[0] > dist[1]]
	x[i, j_closer] = 0
	x[j, i_closer] = 0
	return x, both_nz

def threshold_genes(X,p=99.9,t0=0.05):
	ptile = np.percentile(X,p,axis=1)*t0
	X.T[X.T < ptile] = 0
	return X

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--trainpath', help='Path to directory with composition (phi) and gene module (U) matrices')
	parser.add_argument('--modules', help='Filename of gene module dictionary', default='gene_modules')
	parser.add_argument('--method', help='Signal integration method',default='integrated_intensity')
	parser.add_argument('--region-mask', help='Mask for cells in a region of interest',default=None)
	parser.add_argument('--correct-comeasured',dest='correct_comeasured', action='store_true')
	parser.add_argument('--use-test-idx',dest='use_test', action='store_true')
	parser.add_argument('--save-output',dest='save_output', action='store_true')
	parser.set_defaults(correct_comeasured=False)
	parser.set_defaults(use_test=False)
	parser.set_defaults(save_output=False)
	args,_ = parser.parse_known_args()
	os.environ['KMP_WARNINGS'] = '0'
	f = open('%s/genes.txt' % args.trainpath)
	AllGenes = np.array([line.strip() for line in f])
	f.close()
	phi = np.load('%s/phi.npy' % args.trainpath)
	phi_corr = (np.einsum('ij,ik->jk',phi,phi)/phi.sum(0)).T - np.eye(phi.shape[1])
	relative_abundance = np.load('%s/relative_abundance.npy' % args.trainpath)
	expression_level = np.load('%s/average_on_expression_level.npy' % args.trainpath)
	train_corr = np.load('%s/correlations.npy' % args.trainpath)
	train_corr = distance.squareform(train_corr)
	U = np.load('%s/%s.npy' % (args.trainpath, args.modules))
	# Inliers = np.load('%s/%s.inliers.npy' % (args.trainpath, args.modules),allow_pickle=True)
	if args.use_test:
		test_idx = np.load('%s/%s.test_idx.npy' % (args.trainpath, args.modules))
	else:
		test_idx = np.arange(1e7,dtype=np.int)
	idx_offset = 0
	T = [int(t) for t in args.tissues.split(',')]
	X0 = []
	X1 = []
	X2 = []
	C = []
	for ti,t in enumerate(T):
		tissue = 'tissue%d' % t
		composite_measurements = np.load('%s/FilteredCompositeImages/%s/%s/composite_measurements.npy' % (args.basepath, tissue,args.method))
		direct_measurements = np.load('%s/FilteredCompositeImages/%s/%s/direct_measurements.npy' % (args.basepath, tissue,args.method))
		direct_labels = np.load('%s/FilteredCompositeImages/%s/%s/direct_labels.npy' % (args.basepath, tissue, args.method))
		if args.region_mask is not None:
			CellMasks = load_npz('%s/FilteredCompositeImages/%s/segmented/cell_masks.size_threshold.npz' % (args.basepath,tissue))
			region_mask = imageio.imread('%s/FilteredCompositeImages/%s/stitched_aligned_filtered/%s' % (args.basepath,tissue,args.region_mask))
			cidx = CellMasks.dot(region_mask.flatten())
			cidx = np.where(cidx)[0]
			composite_measurements = composite_measurements[:,cidx]
			direct_measurements = direct_measurements[:,cidx]
		W = sparse_decode_blocks(composite_measurements,phi.dot(U).astype(np.float32),0.1)
		print('Average W entropy: %.2f' % np.average([np.exp(entropy(abs(w))) for w in W.T if w.sum()]))
		xhat = U.dot(W)
		xhat[np.isnan(xhat)] = 0
		xhat[xhat < 0] = 0
		if args.method == 'integrated_intensity_binary':
			xhat = threshold_genes(xhat)
		if args.save_output:
			os.makedirs('%s/ReconstructedImages/%s/%s' % (args.basepath, tissue, args.method), exist_ok=True)
			np.save('%s/ReconstructedImages/%s/%s/segmented.segmentation.npy' % (args.basepath, tissue, args.method), xhat)
		if args.correct_comeasured:
			xhat,cp = select_and_correct_comeasured(xhat,composite_measurements,phi,phi_corr,train_corr)
			np.save('%s/ReconstructedImages/%s/%s/segmented.segmentation.adjusted.npy' % (args.basepath, tissue, args.method), xhat)
		# # exclude outliers for evaluation
		# composite_measurements = composite_measurements[:,Inliers[ti]]
		# direct_measurements = direct_measurements[:,Inliers[ti]]
		# xhat = xhat[:,Inliers[ti]]
		# only evaluate on test data
		test_idx_t = test_idx[(test_idx >= idx_offset)*(test_idx < idx_offset+composite_measurements.shape[1])] - idx_offset
		direct_measurements = direct_measurements[:,test_idx_t]
		xhat = xhat[:,test_idx_t]
		idx_offset += composite_measurements.shape[1]
		X1.append(xhat)
		idx = [np.where(AllGenes == l)[0][0] for l in direct_labels]
		xhat = xhat[idx]
		n = direct_measurements.shape[0]+1
		if args.save_output:
			fig, axes = plt.subplots(max(2,int(np.floor(np.sqrt(n)))), int(np.ceil(np.sqrt(n))))
			axes = axes.flatten()
			plt.rcParams["axes.labelsize"] = 3
		for i in range(n-1):
			corr = 1-distance.correlation(direct_measurements[i],xhat[i])
			if args.save_output:
				sns_plt = sns.scatterplot(direct_measurements[i],xhat[i],ax=axes[i])
				_= sns_plt.set(xlabel='Direct Intensity (arb. units)',ylabel='Recovered Intensity (arb. units)',title='%s; r=%.4f' % (direct_labels[i],corr))
			print(tissue,direct_labels[i],corr)
			C.append((tissue,direct_labels[i],corr))
		if args.save_output:
			corr = 1-distance.correlation(direct_measurements.flatten(),xhat.flatten())
			sns_plt = sns.scatterplot(direct_measurements.flatten(),xhat.flatten(),ax=axes[-1])
			_= sns_plt.set(xlabel='Direct Intensity (arb. units)',ylabel='Recovered Intensity (arb. units)',title='%s; r=%.4f' % ('all points',corr))
			print(tissue,'all points',corr)
			_=plt.tight_layout()
			if args.correct_comeasured:
				outpath = '%s/ReconstructedImages/%s/%s/scatter.segmented.heuristic_correction.png' % (args.basepath, tissue, args.method)
			else:
				outpath = '%s/ReconstructedImages/%s/%s/scatter.segmented.png' % (args.basepath, tissue, args.method)
			fig.savefig(outpath)
			plt.close()
		X0.append(direct_measurements.flatten())
		X2.append(xhat.flatten())
	X0 = np.hstack(X0)
	X1 = np.hstack(X1)
	X2 = np.hstack(X2)
	corr = 1-distance.correlation(X0,X2)
	corr_gene_avg = np.average([c[2] for c in C])
	if args.save_output:
		sns_plt = sns.scatterplot(X0,X2)
		_= sns_plt.set(xlabel='Direct Intensity (arb. units)',ylabel='Recovered Intensity (arb. units)',title='%s; r=%.4f' % ('all tissues, all points',corr))
	print('all tissues, all points',corr)
	print('average gene corr', np.average(corr_gene_avg))
	avg_ng = np.average([np.exp(entropy(x)) for x in X1.T if x.sum()>0])
	print('average genes / cell:',avg_ng)
	if args.save_output:
		fig = sns_plt.get_figure()
		if args.correct_comeasured:
			outpath = '%s/ReconstructedImages/scatter.segmented.heuristic_correction.pdf' % (args.basepath)
		else:
			outpath = '%s/ReconstructedImages/scatter.segmented.pdf' % (args.basepath)
		fig.savefig(outpath)
		plt.close()
		f = open(outpath[:outpath.rfind('/')+1] + 'summary.segmentation.txt','w')
		_=f.write('\t'.join(['Overall correlation', 'Average gene correlation', 'Genes per cell']) + '\n')
		_=f.write('\t'.join([str(x) for x in [corr,corr_gene_avg,avg_ng]]) + '\n')
		_=f.write('\n')
		_=f.write('\t'.join(['Gene','Tissue','Correlation across segmented cells','Expression level in ON cells','Relative abundance in scRNA']) + '\n')
		for c in C:
			el = expression_level[np.where(AllGenes == c[1])[0][0]]
			ra = relative_abundance[np.where(AllGenes == c[1])[0][0]]
			_=f.write('\t'.join([c[1],c[0],str(c[2]),str(el),str(ra)]) + '\n')
		f.close()


	
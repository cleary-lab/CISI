import numpy as np
import argparse
from find_modules import sparse_decode
import pandas as pd
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import entropy

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--trainpath', help='Path to directory with composition (phi) and gene module (U) matrices')
	parser.add_argument('--method', help='Signal integration method',default='integrated_intensity')
	parser.add_argument('--correct-comeasured',dest='correct_comeasured', action='store_true')
	parser.set_defaults(correct_comeasured=False)
	args,_ = parser.parse_known_args()
	sns.set(font_scale=0.5)
	AllGenes = np.load('%s/labels.genes.npy' % args.trainpath)
	phi = np.load('%s/phi.npy' % args.trainpath)
	#phi = (phi.T/phi.sum(1)).T
	phi_corr = (np.einsum('ij,ik->jk',phi,phi)/phi.sum(0)).T - np.eye(phi.shape[1])
	train_corr = np.load('%s/correlations.npy' % args.trainpath)
	train_corr = distance.squareform(train_corr)
	U = np.load('%s/gene_modules.npy' % args.trainpath)
	X0 = []
	X1 = []
	X2 = []
	C = []
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		composite_measurements = np.load('%s/FilteredCompositeImages/%s/%s/composite_measurements.npy' % (args.basepath, tissue,args.method))
		direct_measurements = np.load('%s/FilteredCompositeImages/%s/%s/direct_measurements.npy' % (args.basepath, tissue,args.method))
		direct_labels = np.load('%s/FilteredCompositeImages/%s/%s/direct_labels.npy' % (args.basepath, tissue, args.method))
		W = sparse_decode(composite_measurements,phi.dot(U).astype(np.float32),0.1,method='lasso')
		xhat = U.dot(W)
		xhat[np.isnan(xhat)] = 0
		xhat[xhat < 0] = 0
		if args.correct_comeasured:
			np.save('%s/ReconstructedImages/%s/%s/segmented.segmentation.npy' % (args.basepath, tissue, args.method), xhat)
			xhat,cp = select_and_correct_comeasured(xhat,composite_measurements,phi,phi_corr,train_corr)
			np.save('%s/ReconstructedImages/%s/%s/segmented.segmentation.adjusted.npy' % (args.basepath, tissue, args.method), xhat)
		X1.append(xhat)
		idx = [np.where(AllGenes == l)[0][0] for l in direct_labels]
		xhat = xhat[idx]
		n = direct_measurements.shape[0]+1
		fig, axes = plt.subplots(max(2,int(np.floor(np.sqrt(n)))), int(np.ceil(np.sqrt(n))))
		axes = axes.flatten()
		plt.rcParams["axes.labelsize"] = 3
		for i in range(n-1):
			corr = 1-distance.correlation(direct_measurements[i],xhat[i])
			sns_plt = sns.scatterplot(direct_measurements[i],xhat[i],ax=axes[i])
			_= sns_plt.set(xlabel='Direct Intensity (arb. units)',ylabel='Recovered Intensity (arb. units)',title='%s; r=%.4f' % (direct_labels[i],corr))
			print(tissue,direct_labels[i],corr)
			C.append(corr)
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
	sns_plt = sns.scatterplot(X0,X2)
	_= sns_plt.set(xlabel='Direct Intensity (arb. units)',ylabel='Recovered Intensity (arb. units)',title='%s; r=%.4f' % ('all tissues, all points',corr))
	print('all tissues, all points',corr)
	print('average gene corr', np.average(C))
	fig = sns_plt.get_figure()
	if args.correct_comeasured:
		outpath = '%s/ReconstructedImages/scatter.segmented.heuristic_correction.png' % (args.basepath)
	else:
		outpath = '%s/ReconstructedImages/scatter.segmented.png' % (args.basepath)
	fig.savefig(outpath)
	plt.close()
	# gene-gene correlation
	Corr = 1-distance.squareform(distance.pdist(X1,'correlation')) - np.eye(X1.shape[0])
	df = pd.DataFrame(Corr,columns=AllGenes,index=AllGenes)
	linkage = hc.linkage(distance.squareform(1-Corr-np.eye(X1.shape[0])), method='average')
	sns_plt = sns.clustermap(df,mask=(df == 0), row_linkage=linkage, col_linkage=linkage)
	sns_plt.savefig('%s/ReconstructedImages/gene_similarity.segmented.png' % args.basepath)
	plt.close()
	avg_ng = np.average([np.exp(entropy(x)) for x in X1.T if x.sum()>0])
	print('average genes / cell:',avg_ng)


	
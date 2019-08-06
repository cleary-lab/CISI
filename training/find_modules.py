import numpy as np
import spams
from scipy.stats import entropy
from scipy.spatial import distance
from analyze_predictions import *
import sys
import argparse

THREADS = 20
def smaf(X,d,lda1,lda2,maxItr=10,UW=None,posW=False,posU=True,use_chol=False,module_lower=1,activity_lower=1,donorm=False,mode=1,mink=5,U0=[],U0_delta=0.1,doprint=False):
	# use Cholesky when we expect a very sparse result
	# this tends to happen more on the full vs subsampled matrices
	if UW == None:
		U,W = spams.nmf(np.asfortranarray(X),return_lasso=True,K = d,numThreads=THREADS)
		W = np.asarray(W.todense())
	else:
		U,W = UW
	Xhat = U.dot(W)
	Xnorm = np.linalg.norm(X)**2/X.shape[1]
	for itr in range(maxItr):
		if mode == 1:
			# In this mode the ldas correspond to an approximate desired fit
			# Higher lda will be a worse fit, but will result in a sparser sol'n
			U = spams.lasso(np.asfortranarray(X.T),D=np.asfortranarray(W.T),
			lambda1=lda2*Xnorm,mode=1,numThreads=THREADS,cholesky=use_chol,pos=posU)
			U = np.asarray(U.todense()).T
		elif mode == 2:
			if len(U0) > 0:
				U = projected_grad_desc(W.T,X.T,U.T,U0.T,lda2,U0_delta,maxItr=400)
				U = U.T
			else:
				U = spams.lasso(np.asfortranarray(X.T),D=np.asfortranarray(W.T),
				lambda1=lda2,lambda2=0.0,mode=2,numThreads=THREADS,cholesky=use_chol,pos=posU)
				U = np.asarray(U.todense()).T
		if donorm:
			U = U/np.linalg.norm(U,axis=0)
			U[np.isnan(U)] = 0
		if mode == 1:
			wf = (1 - lda2)
			W = sparse_decode(X,U,lda1,worstFit=wf,mink=mink)
		elif mode == 2:
			if len(U0) > 0:
				W = projected_grad_desc(U,X,W,[],lda1,0.,nonneg=posW,maxItr=400)
			else:
				W = spams.lasso(np.asfortranarray(X),D=np.asfortranarray(U),
				lambda1=lda1,lambda2=1.0,mode=2,numThreads=THREADS,cholesky=use_chol,pos=posW)
				W = np.asarray(W.todense())
		Xhat = U.dot(W)
		module_size = np.average([np.exp(entropy(abs(u))) for u in U.T if u.sum()>0])
		activity_size = np.average([np.exp(entropy(abs(w))) for w in W.T])
		if doprint:
			print(distance.correlation(X.flatten(),Xhat.flatten()),module_size,activity_size,lda1,lda2)
		if module_size < module_lower:
			lda2 /= 2.
		if activity_size < activity_lower:
			lda2 /= 2.
	return U,W

def sparse_decode(Y,D,k,worstFit=1.,mink=0,method='omp',nonneg=False):
	if method == 'omp':
		while k > mink:
			W = spams.omp(np.asfortranarray(Y),np.asfortranarray(D),L=k,numThreads=THREADS)
			W = np.asarray(W.todense())
			fit = 1 - np.linalg.norm(Y - D.dot(W))**2/np.linalg.norm(Y)**2
			if fit < worstFit:
				break
			else:
				k -= 1
	elif method == 'lasso':
		Ynorm = np.linalg.norm(Y)**2/Y.shape[1]
		W = spams.lasso(np.asfortranarray(Y),np.asfortranarray(D),lambda1=k*Ynorm,mode=1,numThreads=THREADS,pos=nonneg)
		W = np.asarray(W.todense())
	return W

def random_phi_subsets_m(m,g,n,d_thresh=0.4):
	Phi = np.zeros((m,g))
	Phi[0,np.random.choice(g,np.random.randint(n[0],n[1]),replace=False)] = 1
	Phi[0] /= Phi[0].sum()
	for i in range(1,m):
		dmax = 1
		while dmax > d_thresh:
			p = np.zeros(g)
			p[np.random.choice(g,np.random.randint(n[0],n[1]),replace=False)] = 1
			p /= p.sum()
			dmax = 1-distance.cdist(Phi[:i],[p],'cosine').min()
		Phi[i] = p
	return Phi

def random_phi_subsets_g(m,g,n,d_thresh=0.4):
	Phi = np.zeros((m,g))
	Phi[np.random.choice(m,np.random.randint(n[0],n[1]),replace=False),0] = 1
	for i in range(1,g):
		dmax = 1
		while dmax > d_thresh:
			p = np.zeros(m)
			p[np.random.choice(m,np.random.randint(n[0],n[1]),replace=False)] = 1
			dmax = 1-distance.cdist(Phi[:,:i].T,[p],'correlation').min()
		Phi[:,i] = p
	Phi = (Phi.T/Phi.sum(1)).T
	return Phi

def check_balance(Phi,thresh=4):
	x = Phi.sum(0) + 1e-7
	if (x.max()/x.min() > thresh) or (Phi.sum(1).min() == 0):
		return False
	else:
		return True

def get_observations(X0,Phi,snr=5,return_noise=False):
	noise = np.array([np.random.randn(X0.shape[1]) for _ in range(X0.shape[0])])
	noise *= np.linalg.norm(X0)/np.linalg.norm(noise)/snr
	if return_noise:
		return Phi.dot(X0 + noise),noise
	else:
		return Phi.dot(X0 + noise)

def compare_results(A,B):
	results = list(correlations(A,B,0))[:-1]
	results += list(compare_distances(A,B))
	results += list(compare_distances(A.T,B.T))
	return results

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--outpath', help="Base directory for saving results")
	parser.add_argument('--all-genes', help="Path to numpy array with all (eg ~10k) gene labels in scRNA matrix")
	parser.add_argument('--select-genes', help="Path to text file with subset of genes to be measured")
	parser.add_argument('--datapath', help="Path to numpy array with expression data (genes x cells)")
	parser.add_argument('--dict-size', help="Number of modules in dictionary", type=int)
	parser.add_argument('--k-sparsity', help="k-sparsity of module activity in each cell", type=int)
	parser.add_argument('--lasso-sparsity', help="Error tolerance applied to l1 sparsity constraint", type=float, default=0.2)
	parser.add_argument('--num-measurements', help="Number of measurements", type=int)
	parser.add_argument('--max-compositions', help="Maximum times each gene is represented (mode G), or max genes per composition (mode M)", type=int)
	parser.add_argument('--mode', help="Mode in which to apply max-compositions", default="G")
	args,_ = parser.parse_known_args()
	L = np.load(args.all_genes)
	X = np.load(args.datapath)
	f = open(args.select_genes)
	Genes = [line.strip() for line in f]
	f.close()
	gidx = [np.where(L == g)[0][0] for g in Genes]
	X = X[gidx]
	X = (X.T/np.linalg.norm(X,axis=1)).T
	train = np.random.choice(np.arange(X.shape[1]),int(X.shape[1]*6/10),replace=False)
	remaining = np.setdiff1d(np.arange(X.shape[1]),train)
	validate = remaining[:int(len(remaining)/2)]
	test = remaining[int(len(remaining)/2):]

	# run SMAF
	U,V = smaf(X[:,train],args.dict_size,args.k_sparsity,args.lasso_sparsity,maxItr=10,use_chol=False,donorm=True,mode=1,mink=0.,doprint=True)
	nz = (U.sum(0) > 0)
	U = U[:,nz]
	print(U.shape)
	np.save('%s/%d_measurements/%d_max/gene_modules.npy' % (args.outpath,args.num_measurements,args.max_composition-1),U)
	if (args.mode == 'M') or (args.mode == 'G'):
		# empirical observation: using a sparsity constraint that is softer than 
		# that used during training slightly improves results
		sparsity = args.lasso_sparsity/10
		# find a good composition matrix by generating a bunch of random matrices
		# and testing each for preservation of distances
		print('%d measurements' % args.num_measurements)
		best = np.zeros(50)
		Phi = [None for _ in best]
		for _ in range(2000):
			while True:
				if args.mode == 'M':
					phi = random_phi_subsets_m(args.num_measurements,X.shape[0],(2,args.max_composition),d_thresh=0.8)
				elif args.mode == 'G':
					phi = random_phi_subsets_g(args.num_measurements,X.shape[0],(1,args.max_composition),d_thresh=0.8)
				if check_balance(phi):
					break
			if _%100 == 0:
				print(_,best)
			y = get_observations(X[:,validate],phi,snr=5)
			w = sparse_decode(y,phi.dot(U),sparsity,method='lasso')
			x2 = U.dot(w)
			r = compare_results(X[:,validate],x2)
			if r[2] > best.min():
				i = np.argmin(best)
				best[i] = r[2]
				Phi[i] = phi
		xs = np.argsort(best)
		best = best[xs[::-1]]
		Phi = [Phi[i] for i in xs]
		d_gene = np.array([np.percentile(1-distance.pdist(phi.dot(U).T,'correlation'),90) for phi in Phi])
		xs = np.argsort(d_gene)
		f2 = open('%s/%d_measurements/%d_max/simulation_results.txt' % (args.outpath,args.num_measurements,args.max_composition-1),'w')
		f2.write('\t'.join(['version','Overall pearson','Overall spearman','Gene average','Sample average','Sample dist pearson','Sample dist spearman','Gene dist pearson','Gene dist spearman','Matrix coherence (90th ptile)']) + '\n')
		for i in xs:
			f1 = open('%s/%d_measurements/%d_max/measurement_compositions/version_%d.txt' % (args.outpath,args.num_measurements,args.max_composition-1,i),'w')
			phi = Phi[i]
			for j in range(args.num_measurements):
				genes = ['channel %d' % j]
				for k,g in enumerate(Genes):
					if phi[j,k] > 0:
						genes.append(g)
					else:
						genes.append('')
				f1.write('\t'.join(genes) + '\n')
			f1.close()
			y = get_observations(X[:,test],phi,snr=5)
			w = sparse_decode(y,phi.dot(U),sparsity,method='lasso')
			x2 = U.dot(w)
			results = compare_results(X[:,test],x2)
			f2.write('\t'.join([str(x) for x in [i]+results+[d_gene[i]]]) + '\n')
			print(d_gene[i],compare_results(X[:,test],x2))
		f2.close()



import numpy as np
from scipy.stats import entropy
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--all-genes', help='Path to all (20k) gene labels')
	parser.add_argument('--selected-genes', help='Path to selected gene labels')
	parser.add_argument('--datapath', help='Path to data')
	parser.add_argument('--savepath', help='Path to save')
	args,_ = parser.parse_known_args()
	X = np.load(args.datapath)
	### old, entropy-based way
	# AllGenes = np.load(args.all_genes)
	# f = open(args.selected_genes)
	# Genes = [line.strip() for line in f]
	# f.close()
	# gidx = [np.where(AllGenes == g)[0][0] for g in Genes]
	# X = X[gidx]
	# effective fraction (shannon diversity) for cells expressing gene
	# Xent = np.array([np.exp(entropy(x)) for x in X])/X.shape[1]
	# average nonzero expression level of each gene
	Xavg = np.array([np.average(x[x>0]) for x in X])
	# # correct for shallow sequencing
	# RA = Xent/(1-np.exp(-Xavg/Xavg.max()*1.25)) # corresponds to a probability of ~29% of observing 0 counts for a gene expressed at Xavg.max() level
	RA = np.average(X > 0,axis=1)
	np.save('%s/relative_abundance.npy' % args.savepath,RA)
	np.save('%s/average_on_expression_level.npy' % args.savepath,Xavg)
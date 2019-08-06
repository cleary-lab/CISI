import numpy as np
import argparse
from scipy.spatial import distance

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--all-genes', help='Path to all (20k) gene labels')
	parser.add_argument('--selected-genes', help='Path to selected gene labels')
	parser.add_argument('--datapath', help='Path to data')
	parser.add_argument('--savepath', help='Path to save')
	args,_ = parser.parse_known_args()
	X = np.load(args.datapath)
	AllGenes = np.load(args.all_genes)
	f = open(args.selected_genes)
	Genes = [line.strip() for line in f]
	f.close()
	gidx = [np.where(AllGenes == g)[0][0] for g in Genes]
	X = X[gidx]
	corr = 1-distance.pdist(X,'correlation')
	np.save('%s/correlations.npy' % args.savepath,corr)
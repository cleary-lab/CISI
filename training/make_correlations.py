import numpy as np
import argparse
from scipy.spatial import distance

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath', help='Path to data (selected genes only)')
	parser.add_argument('--savepath', help='Path to save')
	args,_ = parser.parse_known_args()
	X = np.load(args.datapath)
	corr = 1-distance.pdist(X,'correlation')
	np.save('%s/correlations.npy' % args.savepath,corr)
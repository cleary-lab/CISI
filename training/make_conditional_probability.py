import numpy as np
import argparse

def gene_conditional_probs(X,thresh):
	CP = np.eye(X.shape[0])
	for i in range(X.shape[0]):
		a = (X[i] > thresh)
		for j in range(X.shape[0]):
			b = (X[j] > thresh)
			CP[i,j] = np.average(a*b)/np.average(a)
	return CP

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath', help='Path to data')
	parser.add_argument('--savepath', help='Path to save')
	parser.add_argument('--threshold', help='Expression threshold',type=float,default=10)
	args,_ = parser.parse_known_args()
	X = np.load(args.datapath)
	cond_prob = gene_conditional_probs(X,args.threshold)
	np.save('%s/conditional_probability.npy' % args.savepath,cond_prob)
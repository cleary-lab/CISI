import numpy as np
import os
import argparse
import glob
import imageio
from scipy.sparse import csr_matrix, save_npz


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	args,_ = parser.parse_known_args()
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		data = []
		row_ind = []
		col_ind = []
		FP = glob.glob(os.path.join(args.basepath,tissue,'segmented','ExpandedNuclei_*'))
		for fp in FP:
			cell_num = int(fp.split('_')[-1].split('.')[0])-1
			im = imageio.imread(fp)
			mask = np.where(im.flatten())[0]
			data.append(np.ones(len(mask),dtype=np.float32))
			row_ind.append(np.ones(len(mask),dtype=np.int)*cell_num)
			col_ind.append(mask)
		data = np.hstack(data)
		row_ind = np.hstack(row_ind)
		col_ind = np.hstack(col_ind)
		X = csr_matrix((data,(row_ind,col_ind)),shape=[len(FP),len(im.flatten())],dtype=np.float32)
		save_npz('%s/%s/segmented/cell_masks.npz' % (args.basepath,tissue),X)
		print(tissue)



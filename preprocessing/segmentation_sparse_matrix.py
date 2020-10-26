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
	parser.add_argument('--mask-file', help='Mask file (or prefix), eg ExpandedNuclei_* or DAPI.tiff', default='All_Composite.tiff')
	args,_ = parser.parse_known_args()
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		if '*' in args.mask_file:
			# separate file for each cell
			data = []
			row_ind = []
			col_ind = []
			FP = glob.glob(os.path.join(args.basepath,tissue,'segmented',args.mask_file))
			for fp in FP:
				cell_num = int(fp.split('_')[-1].split('.')[0])-1
				im = imageio.imread(fp).flatten()
				mask = np.where(im)[0]
				data.append(np.ones(len(mask),dtype=np.float32))
				row_ind.append(np.ones(len(mask),dtype=np.int)*cell_num)
				col_ind.append(mask)
			data = np.hstack(data)
			row_ind = np.hstack(row_ind)
			col_ind = np.hstack(col_ind)
		else:
			im = imageio.imread('%s/%s/segmented/%s' % (args.basepath,tissue,args.mask_file)).flatten()
			col_ind = np.where(im > im.min())[0]
			data = np.ones(len(col_ind),dtype=np.float32)
			if im.dtype not in (np.dtype('float32'), np.dtype('float64')):
				row_ind = im[col_ind].astype(np.int)-1
			else:
				# values may have overflowed to floats, but are still unique
				row_ind = []
				mask_dict = {}
				for i in im[col_ind]:
					if i not in mask_dict:
						mask_dict[i] = len(mask_dict)
					row_ind.append(mask_dict[i])
				row_ind = np.array(row_ind)
		X = csr_matrix((data,(row_ind,col_ind)),shape=[len(np.unique(row_ind)),len(im)],dtype=np.float32)
		save_npz('%s/%s/segmented/cell_masks.npz' % (args.basepath,tissue),X)
		print(tissue,X.shape,len(data))



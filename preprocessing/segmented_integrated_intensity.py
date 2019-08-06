import numpy as np
import os
import argparse
import glob
import imageio
from scipy.sparse import load_npz


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--stitched-subdir', help='Subdirectory with stitched images', default='stitched_aligned_filtered')
	parser.add_argument('--mask-basepath', help='Optional alternative basepath for cell masks', default=None)
	parser.add_argument('--as-average', dest='as_average', action='store_true')
	parser.set_defaults(as_average=False)
	args,_ = parser.parse_known_args()
	if args.mask_basepath is None:
		args.mask_basepath = args.basepath
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		CellMasks = load_npz('%s/%s/segmented/cell_masks.npz' % (args.mask_basepath,tissue))
		if args.as_average:
			for i in range(CellMasks.shape[0]):
				ii = CellMasks.indptr[i]
				ij = CellMasks.indptr[i+1]
				CellMasks.data[ii:ij] = 1/(ij-ii)
		FP = glob.glob(os.path.join(args.basepath,tissue,args.stitched_subdir,'*.tiff'))
		FP_composites = [fp for fp in FP if 'Composite_' in fp.split('/')[-1]]
		FP_direct = [fp for fp in FP if (('Composite_' not in fp.split('/')[-1]) and ('DAPI' not in fp.split('/')[-1]) and ('merged' not in fp.split('/')[-1]))]
		# sort the composites by number
		FP_composites = [(int(fp.split('_')[-1].split('.')[0]),fp) for fp in FP_composites]
		FP_composites = [fp[1] for fp in sorted(FP_composites)]
		X = []
		L = []
		for fp in FP_composites:
			im = imageio.imread(fp).flatten()
			X.append(CellMasks.dot(im))
			L.append(fp.split('/')[-1].split('.')[0])
		if args.as_average:
			_=os.system('mkdir %s/%s/average_intensity' % (args.basepath,tissue))
			np.save('%s/%s/average_intensity/composite_measurements.npy' % (args.basepath,tissue),X)
			np.save('%s/%s/average_intensity/composite_labels.npy' % (args.basepath,tissue),L)
		else:
			_=os.system('mkdir %s/%s/integrated_intensity' % (args.basepath,tissue))
			np.save('%s/%s/integrated_intensity/composite_measurements.npy' % (args.basepath,tissue),X)
			np.save('%s/%s/integrated_intensity/composite_labels.npy' % (args.basepath,tissue),L)
		X = []
		L = []
		# get the size of cell masks
		FP_mask = glob.glob(os.path.join(args.mask_basepath,tissue,'segmented','ExpandedNuclei_*'))
		n = imageio.imread(FP_mask[0]).shape
		for fp in FP_direct:
			im = imageio.imread(fp)
			# images might have been padded on the bottom and right
			im = im[:n[0],:n[1]].flatten()
			X.append(CellMasks.dot(im))
			L.append('.'.join(fp.split('/')[-1].split('.')[:-1]))
		if args.as_average:
			np.save('%s/%s/average_intensity/direct_measurements.npy' % (args.basepath,tissue),X)
			np.save('%s/%s/average_intensity/direct_labels.npy' % (args.basepath,tissue),L)
		else:
			np.save('%s/%s/integrated_intensity/direct_measurements.npy' % (args.basepath,tissue),X)
			np.save('%s/%s/integrated_intensity/direct_labels.npy' % (args.basepath,tissue),L)
		print(tissue)
			




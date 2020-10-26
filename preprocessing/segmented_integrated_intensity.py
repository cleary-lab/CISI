import numpy as np
import os
import argparse
import glob
import imageio
from scipy.sparse import load_npz, save_npz
from sklearn.mixture import GaussianMixture

def threshold_noise(Y,K=6):
	nonzero_thresh = np.percentile(Y,1,axis=1)*1.1 + 1
	T = np.zeros(Y.shape[0])
	for j in range(Y.shape[0]):
		Models = []
		for i in range(2,K+1):
			model = GaussianMixture(n_components=i)
			_=model.fit(Y[j,Y[j]>nonzero_thresh[j]].reshape(-1,1))
			Models.append(model)
		bic = np.array([model.bic(Y[j].reshape(-1,1)) for model in Models])
		k=bic.argmin()
		# if k > 0:
		l = Models[k].predict(Y[j].reshape(-1,1))
		thresh = Y[j,l==Models[k].means_.argmin(0)].max()
		T[j] = thresh
	return T

def find_inliers(composite_measurements,direct_measurements,CellMasks,pixel_size,min_cell_area,max_cell_area):
	X = np.vstack([composite_measurements,direct_measurements])
	thresh = np.percentile(X,99.9,axis=1)*2
	intensity_inliers = ((X.T < thresh).sum(1) == X.shape[0])
	cell_sizes = np.array(CellMasks.sum(1))[:,0]*pixel_size**2
	include_size_min = cell_sizes >= min_cell_area
	include_size_max = cell_sizes <= max_cell_area
	size_inliers = include_size_max*include_size_min
	cidx = np.where(intensity_inliers*size_inliers)[0]
	return cidx

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--stitched-subdir', help='Subdirectory with stitched images', default='stitched_aligned_filtered')
	parser.add_argument('--mask-basepath', help='Optional alternative basepath for cell masks', default=None)
	parser.add_argument('--pixel-size', help='Pixel size in microns',type=float,default=0.323370542)
	parser.add_argument('--min-cell-area', help='Minimum area threshold for cells (in microns^2)',type=float,default=25)
	parser.add_argument('--max-cell-area', help='Maximum area threshold for cells (in microns^2)',type=float,default=500)
	parser.add_argument('--as-average', dest='as_average', action='store_true')
	parser.add_argument('--binary', dest='binary', action='store_true')
	parser.add_argument('--log', dest='log', action='store_true')
	parser.set_defaults(as_average=False)
	parser.set_defaults(binary=False)
	parser.set_defaults(log=False)
	parser.set_defaults(mixture_model=False)
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
		FP_direct = [fp for fp in FP if (('Composite_' not in fp.split('/')[-1]) and ('All_Composite' not in fp.split('/')[-1]) and ('DAPI' not in fp.split('/')[-1]) and ('merged' not in fp.split('/')[-1]))]
		# sort the composites by number
		FP_composites = [(int(fp.split('_')[-1].split('.')[0]),fp) for fp in FP_composites]
		FP_composites = [fp[1] for fp in sorted(FP_composites)]
		X = []
		L = []
		for fp in FP_composites:
			im = imageio.imread(fp).flatten()
			if args.binary:
				im = (im > 0).astype(im.dtype)
			if args.log:
				im = np.log(im + 1).astype(im.dtype)
			X.append(CellMasks.dot(im))
			L.append(fp.split('/')[-1].split('.')[0])
		if args.as_average:
			if args.binary:
				savepath = '%s/%s/average_intensity_binary' % (args.basepath,tissue)
			elif args.log:
				savepath = '%s/%s/average_intensity_log' % (args.basepath,tissue)
			else:
				savepath = '%s/%s/average_intensity' % (args.basepath,tissue)
		else:
			if args.binary:
				savepath = '%s/%s/integrated_intensity_binary' % (args.basepath,tissue)
			elif args.log:
				savepath = '%s/%s/integrated_intensity_log' % (args.basepath,tissue)
			else:
				savepath = '%s/%s/integrated_intensity' % (args.basepath,tissue)
		_=os.system('mkdir %s' % savepath)
		X = np.array(X)
		np.save('%s/composite_labels.npy' % savepath,L)
		X_direct = []
		L = []
		# get the size of cell masks
		FP_mask = glob.glob(os.path.join(args.mask_basepath,tissue,'segmented','ExpandedNuclei_*'))
		if len(FP_mask) == 0:
			FP_mask = glob.glob(os.path.join(args.mask_basepath,tissue,'segmented','All_Composite.tiff'))
		n = imageio.imread(FP_mask[0]).shape
		for fp in FP_direct:
			im = imageio.imread(fp)
			# images might have been padded on the bottom and right
			im = im[:n[0],:n[1]].flatten()
			if args.binary:
				im = (im > 0).astype(im.dtype)
			elif args.log:
				im = np.log(im + 1).astype(im.dtype)
			X_direct.append(CellMasks.dot(im))
			L.append('.'.join(fp.split('/')[-1].split('.')[:-1]))
		if (tissue == 'tissue2') and ('Foxp2' in L): # Foxp2 was measured twice in tissue2
			gi = L.index('Foxp2')
			X_direct.append(X_direct[gi])
			L.append(L[gi])
		if (tissue == 'tissue7'): # Pdgfra and Prox1 validation did not appear to work (Pdgfra looks good in tissue12)
			gi = L.index('Pdgfra')
			X_direct = X_direct[:gi] + X_direct[gi+1:]
			L = L[:gi] + L[gi+1:]
			gi = L.index('Prox1')
			X_direct = X_direct[:gi] + X_direct[gi+1:]
			L = L[:gi] + L[gi+1:]
			gi = L.index('Klf4')
			X_direct.append(X_direct[gi])
			L.append(L[gi])
			gi = L.index('Fgf13')
			X_direct.append(X_direct[gi])
			L.append(L[gi])
		np.save('%s/direct_labels.npy' % savepath,L)
		X_direct = np.array(X_direct)
		include = find_inliers(X,X_direct,CellMasks,args.pixel_size,args.min_cell_area,args.max_cell_area)
		print('%s, %d total cells, %d pass size thresholds' % (tissue, CellMasks.shape[0], include.shape[0]))
		CellMasks = CellMasks[include]
		if not args.binary and not args.log:
			save_npz('%s/%s/segmented/cell_masks.size_threshold.npz' % (args.mask_basepath,tissue), CellMasks)
		X = X[:,include]
		np.save('%s/composite_measurements.npy' % savepath,X)
		X_direct = X_direct[:,include]
		np.save('%s/direct_measurements.npy' % savepath,X_direct)
		T = threshold_noise(X_direct,K=2)
		print('direct thresholds:')
		print(L)
		print(T)
		print((X_direct.T > T).sum(0))
		np.save('%s/direct_thresholds.npy' % savepath,T)
		print(tissue)

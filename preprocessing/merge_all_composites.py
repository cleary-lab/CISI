import numpy as np
import os
import argparse
import glob
import imageio
from scipy import ndimage

def filter(im, filter_size):
	if filter_size > 0:
		if len(im.shape) == 3:
			for i in range(im.shape[2]):
				im[:,:,i] = ndimage.gaussian_filter(im[:,:,i], filter_size)
		else:
			im = ndimage.gaussian_filter(im, filter_size)
	return im

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--filter-size', help='Size of gaussian filter',type=float,default=4)
	parser.add_argument('--stitched-subdir', help='Subdirectory with stitched images', default='stitched_aligned_filtered')
	args,_ = parser.parse_known_args()
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		FP = glob.glob(os.path.join(args.basepath,tissue,args.stitched_subdir,'Composite_*'))
		im = imageio.imread(FP[0])
		for fp in FP[1:]:
			im = np.maximum(im,imageio.imread(fp))
		im = filter(im, args.filter_size)
		imageio.imwrite('%s/%s/%s/All_Composite.tiff' % (args.basepath,tissue,args.stitched_subdir),im)
		print(tissue)

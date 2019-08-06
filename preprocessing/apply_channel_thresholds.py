import numpy as np
import os
import argparse
import glob
import imageio

def apply_threshold(im,x0,x1):
	dt = im.dtype
	max_value = np.iinfo(dt).max
	im[im < x0] = x0
	im[im > x1] = x1
	im = im - x0
	im = np.rint(im/(x1 - x0)*max_value).astype(dt)
	return im	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--thresholds-file', help='File containing channel min and max thresholds to apply to all images')
	parser.add_argument('--zero-width', help='Set (width) edge pixels to zero',type=int,default=6)
	parser.add_argument('--keep-existing', help='Keep existing stitched files (if present)',dest='keep_existing', action='store_true')
	parser.set_defaults(keep_existing=False)
	args,_ = parser.parse_known_args()
	f = open(args.thresholds_file)
	thresholds = [line.strip().split() for line in f]
	f.close()
	thresholds = [[int(t[0]),int(t[1])] for t in thresholds]
	FP = glob.glob(os.path.join(args.basepath,'*','*','stitched.background_subtract.tiff'))
	for fp in FP:
		outfile = fp.replace('.background_subtract.tiff','.threshold.tiff')
		if (args.keep_existing and not os.path.isfile(outfile)) or (not args.keep_existing):
			im = imageio.imread(fp)
			im[:args.zero_width,:] = 0
			im[-args.zero_width:,:] = 0
			im[:,:args.zero_width] = 0
			im[:,-args.zero_width:] = 0
			new_im = np.zeros(im.shape,dtype=im.dtype)
			for i in range(im.shape[-1]):
				im_i = apply_threshold(im[:,:,i],thresholds[i][0],thresholds[i][1])
				new_im[:,:,i] = im_i
			imageio.imwrite(outfile,new_im)
			print(fp)
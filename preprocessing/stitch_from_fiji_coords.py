import numpy as np
import os
import argparse
import glob
from ast import literal_eval
import imageio
from scipy.ndimage import fourier_shift
from scipy import ndimage

def filter(im, filter_size):
	if filter_size > 0:
		if len(im.shape) == 3:
			for i in range(im.shape[2]):
				im[:,:,i] = ndimage.median_filter(im[:,:,i], filter_size)
	return im

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--channels', help='Comma separated ordered list of channels, e.g. Conf-488,conf-561,conf-640,conf-405')
	parser.add_argument('--filter-size', help='Size of median filter',type=int,default=8)
	parser.add_argument('--jobnum', help='Job index for running on cluster',type=int,default=-1)
	parser.add_argument('--blank-round', help='Round in which blank images were acquired (will use bigger filter on these)')
	parser.add_argument('--keep-existing', help='Keep existing stitched files (if present)',dest='keep_existing', action='store_true')
	parser.set_defaults(keep_existing=False)
	args,_ = parser.parse_known_args()
	channels = args.channels.split(',')
	FP = glob.glob(os.path.join(args.basepath,'*','*','DAPI.conf-405','TileConfiguration.registered.txt'))
	if args.jobnum > -1:
		FP = FP[args.jobnum:args.jobnum+1]
	for fp in FP:
		outfile = '/'.join(fp.split('/')[:-2] + ['stitched.tiff'])
		if (args.keep_existing and not os.path.isfile(outfile)) or (not args.keep_existing):
			fovs = []
			coords = []
			f = open(fp)
			header = [f.readline() for _ in range(4)]
			for line in f:
				ls = line.strip().split(';')
				fovs.append(ls[0])
				coords.append(literal_eval(ls[2].strip()))
			f.close()
			coords = np.rint(coords).astype(np.int)
			coords[:,0] -= coords[:,0].min()
			coords[:,1] -= coords[:,1].min()
			fov_size = imageio.imread(fp.replace('TileConfiguration.registered.txt',fovs[0])).shape
			stitched_size = (coords.max(0) + fov_size)[::-1]
			stitched_size = np.hstack([stitched_size,len(channels)])
			stitched = np.zeros(stitched_size,dtype=np.uint16)
			for c,channel in enumerate(channels):
				for fov,coord in zip(fovs,coords):
					fp_channel_fov = glob.glob(os.path.join('/'.join(fp.split('/')[:-2] + ['*.%s' % channel,fov])))
					if len(fp_channel_fov) == 1:
						stitched[coord[1]:coord[1]+fov_size[1],coord[0]:coord[0]+fov_size[0],c] = imageio.imread(fp_channel_fov[0])
			if fp.split('/')[-3] == args.blank_round:
				stitched = filter(stitched, args.filter_size*2)
			else:
				stitched = filter(stitched, args.filter_size)
			imageio.imwrite(outfile, stitched)
			print(outfile)
import numpy as np
import os
import argparse
import glob
from scipy.ndimage import gaussian_filter
from ast import literal_eval
import imageio

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--blank-round', help='Round in which blank images were collected (skip these)')
	parser.add_argument('--size', help='Pixels along each dimension of one fov',type=int,default=2048)
	parser.add_argument('--filter-width', help='Width of flat field gaussian_filter; eg 16 -> 1/16th of image size',type=int,default=8)
	parser.add_argument('--existing-flat-field', help='Path to existing flat field',default=None)
	args,_ = parser.parse_known_args()
	FP_blanks = glob.glob(os.path.join(args.basepath,'*',args.blank_round,'DAPI.conf-405','TileConfiguration.registered.txt'))
	FP = glob.glob(os.path.join(args.basepath,'*','*','DAPI.conf-405','TileConfiguration.registered.txt'))
	FP = [fp for fp in FP if fp not in FP_blanks]
	if args.existing_flat_field is None:
		flat_field = np.zeros((args.size,args.size))
		fov_count = 0
		for fp in FP:
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
			im = imageio.imread(fp.replace('DAPI.conf-405/TileConfiguration.registered.txt','stitched.threshold.tiff'))
			im = np.average(im[:,:,:3],axis=2)
			for coord in coords:
				coord[0] = min(im.shape[1]-args.size,coord[0])
				coord[1] = min(im.shape[0]-args.size,coord[1])
				flat_field += im[coord[1]:coord[1]+args.size, coord[0]:coord[0]+args.size]
				fov_count += 1
		flat_field /= fov_count
		flat_field = gaussian_filter(flat_field, args.size//args.filter_width)
		print('done calculating flat field')
		np.save('%s/flat_field.npy' % args.basepath,flat_field)
	else:
		flat_field = np.load(args.existing_flat_field)
	inverse_field = 1/flat_field
	inverse_field /= inverse_field.max()
	for fp in FP:
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
		im = imageio.imread(fp.replace('DAPI.conf-405/TileConfiguration.registered.txt','stitched.threshold.tiff'))
		im_corrected = np.zeros(im.shape)
		for coord in coords:
			coord[0] = min(im.shape[1]-args.size,coord[0])
			coord[1] = min(im.shape[0]-args.size,coord[1])
			x = np.transpose(im[coord[1]:coord[1]+args.size, coord[0]:coord[0]+args.size],axes=[2,0,1])*inverse_field
			im_corrected[coord[1]:coord[1]+args.size, coord[0]:coord[0]+args.size] = np.transpose(x,axes=[1,2,0])
		dt = im.dtype
		max_scale = np.iinfo(dt).max
		#for i in range(im_corrected.shape[2]):
		#	im_corrected[:,:,i] = np.rint(im_corrected[:,:,i]/im_max*max_scale)
		im_corrected = np.rint(im_corrected).astype(dt)
		outpath = fp.replace('DAPI.conf-405/TileConfiguration.registered.txt','stitched.threshold.corrected.tiff')
		imageio.imwrite(outpath,im_corrected)
		print(outpath)



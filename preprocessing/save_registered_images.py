import numpy as np
import os
import argparse
import glob
from collections import defaultdict
from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage import fourier_shift
import imageio

def parse_rounds(fp):
	Rounds2Channels = defaultdict(list)
	f = open(fp,encoding='utf-8')
	header = f.readline()
	for line in f:
		ls = line.strip().split(',')
		Rounds2Channels[(ls[0],ls[1])].append(ls[2])
	f.close()
	return Rounds2Channels

def find_all_shifts(ref_image,other_images):
	shifts = []
	for im2 in other_images:
		shifts.append(find_shift(ref_image,im2))
	return shifts

def find_shift(im1,im2,scale=0.25):
	im1_r = rescale(im1,scale)
	im2_r = rescale(im2,scale)
	shift, error, diffphase = register_translation(im1_r, im2_r)
	return shift/scale

def apply_shifts(im,shift):
	offset_image = []
	for i in range(im.shape[2]):
		offset_im = fourier_shift(np.fft.fftn(im[:,:,i]), shift)
		offset_image.append(np.fft.ifftn(offset_im))
	offset_image = np.transpose(offset_image,axes=[1,2,0]).astype(np.float32)
	return offset_image

def crop_all_shifted(images,shifts):
	min_s = np.min(shifts,axis=0).astype(np.int)
	max_s = np.max(shifts,axis=0).astype(np.int)
	if min_s[0] < 0:
		images = images[:min_s[0]]
	if max_s[0] > 0:
		images = images[max_s[0]:]
	if min_s[1] < 0:
		images = images[:,:min_s[1]]
	if max_s[1] > 0:
		images = images[:,max_s[1]:]
	return images

def normalize_image_scale(im,max_value,thresh_each=False):
	if thresh_each:
		for i in range(im.shape[-1]):
			thresh = np.percentile(im,max_value)
			im[:,:,i][im[:,:,i] > thresh] = thresh
			im[:,:,i] = im[:,:,i]/thresh
		return im.astype(np.float32)
	else:
		return (im/max_value).astype(np.float32)

def break_into_tiles(im,tile_size):
	tiles = []
	tile_pattern = []
	ti = 0
	for i in range(0,im.shape[0],tile_size):
		tile_row = []
		for j in range(0,im.shape[1],tile_size):
			x = im[i:i+tile_size, j:j+tile_size]
			x = np.pad(x, [(0,tile_size-x.shape[0]), (0,tile_size-x.shape[1]), (0,0)], 'constant')
			tiles.append(x)
			tile_row.append(ti)
			ti += 1
		if len(tile_row) > 0:
			tile_pattern.append(tile_row)
	return tiles,tile_pattern

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--tile-size', help='Size of output images',type=int,default=1024)
	parser.add_argument('--rounds-to-channels', help='CSV list of tissue, round, composition labels')
	parser.add_argument('--dapi-index', help='Channel index for DAPI images',type=int,default=3)
	parser.add_argument('--processed-filename', help='Name of processed files',default='stitched.threshold.corrected.tiff')
	parser.add_argument('--save-tiles', dest='save_tiles', action='store_true')
	parser.add_argument('--save-stitched', dest='save_stitched', action='store_true')
	parser.set_defaults(save_tiles=False)
	parser.set_defaults(save_stitched=False)
	args,_ = parser.parse_known_args()
	Rounds2Channels = parse_rounds(args.rounds_to_channels)
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		FP = glob.glob(os.path.join('%s/%s/round*/%s' % (args.basepath,tissue,args.processed_filename)))
		FP = sorted(FP)
		Images = [imageio.imread(fp) for fp in FP]
		min_shape = (min(im.shape[0] for im in Images), min(im.shape[1] for im in Images))
		Images = [im[:min_shape[0],:min_shape[1]] for im in Images]
		Images1 = [im[:,:,args.dapi_index] for im in Images]
		shifts = find_all_shifts(Images1[0],Images1[1:])
		ImagesAligned = []
		for i,im in enumerate(Images):
			if i == 0:
				ImagesAligned.append(im.astype(np.float32))
			else:
				ImagesAligned.append(apply_shifts(im,shifts[i-1]))
		ImagesAligned = [crop_all_shifted(images,shifts) for images in ImagesAligned]
		if args.save_stitched:
			_=os.system('mkdir %s/%s/stitched_aligned_filtered' % (args.basepath,tissue))
			for images,fp in zip(ImagesAligned,FP):
				chan_idx = np.where(images.reshape([-1,images.shape[-1]]).sum(0) > 0)[0]
				im = images[:,:,chan_idx]
				round = fp.split('/')[-2]
				channels = Rounds2Channels[(tissue,round)]
				if (round == 'round1') and ('DAPI' not in channels):
					channels.insert(args.dapi_index,'DAPI')
				for i,c in enumerate(channels):
					imageio.imwrite('%s/%s/stitched_aligned_filtered/%s.tiff' % (args.basepath,tissue,c), np.rint(im[:,:,i]).astype(Images[0].dtype))
		if args.save_tiles:
			_=os.system('mkdir %s/%s/arrays_aligned_filtered' % (args.basepath,tissue))
			for images,fp in zip(ImagesAligned,FP):
				max_value = np.iinfo(Images[0].dtype).max
				images = normalize_image_scale(images,max_value)
				chan_idx = np.where(images.reshape([-1,images.shape[-1]]).sum(0) > 0)[0]
				im = images[:,:,chan_idx]
				round = fp.split('/')[-2]
				channels = Rounds2Channels[(tissue,round)]
				if (round == 'round1') and ('DAPI' not in channels):
					channels.insert(args.dapi_index,'DAPI')
				tiles, new_fov_pattern = break_into_tiles(im,args.tile_size)
				new_fov_pattern = np.array(new_fov_pattern)
				for tile, new_fov in zip(tiles,new_fov_pattern.flatten()):
					for i,c in enumerate(channels):
						new_path = '%s/%s/arrays_aligned_filtered/fov_%d.%s.npy' % (args.basepath,tissue,new_fov,c)
						np.save(new_path,tile[:,:,i])
				np.save('%s/%s/modified_fov_pattern.npy' % (args.basepath,tissue), new_fov_pattern)
		print(tissue)
	
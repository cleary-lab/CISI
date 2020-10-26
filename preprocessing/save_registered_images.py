import numpy as np
import os
import argparse
import glob
from collections import defaultdict
from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage import fourier_shift
from scipy import signal
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

def find_and_apply_shifts(ref_image,other_images,dapi_index,block_size=500,block_pad=250):
	im1 = ref_image[:,:,dapi_index]
	ImagesAligned = [ref_image.astype(np.float32)]
	for im2 in other_images:
		im2_shift = np.zeros(im2.shape,dtype=np.float32)
		for i in range(0,im1.shape[0],block_size):
			for j in range(0,im1.shape[1],block_size):
				im1_block = im1[max(0,i-block_pad):i+block_size+block_pad,max(0,j-block_pad):j+block_size+block_pad]
				im2_block = im2[i:i+block_size,j:j+block_size]
				if (im1_block.sum() > 0) and (im2_block.sum() > 0):
					pad0 = im1_block.shape[0] - im2_block.shape[0]
					pad1 = im1_block.shape[1] - im2_block.shape[1]
					pad0 = (int(pad0/2), int(pad0/2 + pad0%2))
					pad1 = (int(pad1/2), int(pad1/2 + pad1%2))
					im2_block = np.pad(im2_block, (pad0, pad1, (0,0)), 'constant').astype(im1_block.dtype)
					shift, error, diffphase = register_translation(im1_block, im2_block[:,:,dapi_index])
					offset_image = apply_shifts(im2_block,shift)
					im2_shift[max(0,i-block_pad):i+block_size+block_pad,max(0,j-block_pad):j+block_size+block_pad] += offset_image
		ImagesAligned.append(im2_shift)
	return ImagesAligned

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

def rescale_intensities(Images,max_value,dapi_index,eps=1e-4, ptile=97):
	# rescale all the measurements in each channel according to the min,max in that channel
	Images = np.array(Images)
	num_channels = Images.shape[-1]
	I = Images.reshape([-1,num_channels])
	channel_min = np.zeros(num_channels)
	channel_max = np.zeros(num_channels)
	for i in range(num_channels):
		if i == dapi_index:
			channel_min[i] = 0
			channel_max[i] = I[:,i].max()
		else:
			channel_min[i] = np.percentile(I[:,i][I[:,i] > eps], 100-ptile)
			channel_max[i] = np.percentile(I[:,i][I[:,i] > eps], ptile)
	print(channel_min)
	print(channel_max)
	Images = (Images - channel_min)/channel_max
	Images[Images < 0] = 0
	Images[Images > 1] = 1
	return Images*max_value

def get_kernel(sigma):
	x = np.linspace(-sigma*3,sigma*3,sigma*6+1)
	x = np.exp(-x**2/sigma**2)
	x /= np.trapz(x)
	return x[:,np.newaxis]*x[np.newaxis,:]

def normalize_dapi_intensity(Images,dapi_index,sigma1=600,sigma2=100):
	K = get_kernel(sigma1)
	im0 = signal.fftconvolve(Images[0][:,:,dapi_index], K, mode='same').T + 1
	K = get_kernel(sigma2)
	Images_new = []
	for im in Images:
		im_blur = signal.fftconvolve(im[:,:,dapi_index], K, mode='same').T + 1
		im_new = im.T*im0/im_blur
		Images_new.append(im_new.T.astype(np.float32))
	return Images_new

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--tile-size', help='Size of output images',type=int,default=256)
	parser.add_argument('--rounds-to-channels', help='CSV list of tissue, round, composition labels')
	parser.add_argument('--dapi-index', help='Channel index for DAPI images',type=int,default=3)
	parser.add_argument('--processed-filename', help='Name of processed files, or eg DAPI.tiff if separated by channel',default='DAPI.tiff')
	parser.add_argument('--tmppath', help='Tmp path to store arrays_aligned_filtered',default=None)
	parser.add_argument('--rescale-intensity',help='Rescale pixel intensities according the (nonzero) 1st,99th ptile observed in each channel', dest='rescale_intensity', action='store_true')
	parser.add_argument('--normalize-dapi', dest='normalize_dapi', action='store_true')
	parser.add_argument('--save-tiles', dest='save_tiles', action='store_true')
	parser.add_argument('--save-stitched', dest='save_stitched', action='store_true')
	parser.add_argument('--shift-blocks',help='Find shifts in blocks. Used when stitching has failed in some round.', dest='shift_blocks', action='store_true')
	parser.set_defaults(rescale_intensity=False)
	parser.set_defaults(normalize_dapi=False)
	parser.set_defaults(save_tiles=False)
	parser.set_defaults(save_stitched=False)
	parser.set_defaults(shift_blocks=False)
	args,_ = parser.parse_known_args()
	Rounds2Channels = parse_rounds(args.rounds_to_channels)
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		FP = glob.glob(os.path.join('%s/%s/round*/%s' % (args.basepath,tissue,args.processed_filename)))
		FP = sorted(FP)
		Images = [imageio.imread(fp) for fp in FP]
		if 'DAPI' in args.processed_filename:
			for i in range(len(FP)):
				round = FP[i].split('/')[-2]
				channels = Rounds2Channels[(tissue,round)]
				im = [imageio.imread(FP[i].replace('DAPI',channel)) for channel in channels]
				im.insert(args.dapi_index,Images[i])
				Images[i] = np.transpose(im,axes=(1,2,0))
		min_shape = (min(im.shape[0] for im in Images), min(im.shape[1] for im in Images))
		Images = [im[:min_shape[0],:min_shape[1]] for im in Images]
		if args.shift_blocks:
			ImagesAligned = find_and_apply_shifts(Images[0],Images[1:],args.dapi_index)
		else:
			Images1 = [im[:,:,args.dapi_index] for im in Images]
			shifts = find_all_shifts(Images1[0],Images1[1:])
			ImagesAligned = []
			for i,im in enumerate(Images):
				if i == 0:
					ImagesAligned.append(im.astype(np.float32))
				else:
					ImagesAligned.append(apply_shifts(im,shifts[i-1]))
			ImagesAligned = [crop_all_shifted(images,shifts) for images in ImagesAligned]
		if args.normalize_dapi:
			ImagesAligned = normalize_dapi_intensity(ImagesAligned,args.dapi_index)
		max_value = np.iinfo(Images[0].dtype).max
		if args.rescale_intensity:
			ImagesAligned = rescale_intensities(ImagesAligned,max_value,args.dapi_index)
		if args.save_stitched:
			_=os.system('mkdir %s/%s/stitched_aligned_filtered' % (args.basepath,tissue))
			for images,fp in zip(ImagesAligned,FP):
				chan_idx = np.where(images.reshape([-1,images.shape[-1]]).sum(0) > 0)[0]
				im = images[:,:,chan_idx]
				round = fp.split('/')[-2]
				channels = Rounds2Channels[(tissue,round)]
				channels.insert(args.dapi_index,'DAPI_%s' % round)
				for i,c in enumerate(channels):
					imageio.imwrite('%s/%s/stitched_aligned_filtered/%s.tiff' % (args.basepath,tissue,c), np.rint(im[:,:,i]).astype(Images[0].dtype))
		if args.save_tiles:
			if args.tmppath is None:
				array_path = args.basepath
			else:
				array_path = args.tmppath
				_=os.system('mkdir %s/%s' % (array_path,tissue))
			_=os.system('mkdir %s/%s/arrays_aligned_filtered' % (array_path,tissue))
			for images,fp in zip(ImagesAligned,FP):
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
						#new_path = '%s/%s/arrays_aligned_filtered/fov_%d.%s.npy' % (args.basepath,tissue,new_fov,c)
						#np.save(new_path,tile[:,:,i])
						new_path = '%s/%s/arrays_aligned_filtered/fov_%d.%s.tiff' % (array_path,tissue,new_fov,c)
						imageio.imwrite(new_path,tile[:,:,i])
				np.save('%s/%s/modified_fov_pattern.npy' % (args.basepath,tissue), new_fov_pattern)
		print(tissue)


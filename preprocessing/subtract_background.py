import numpy as np
import os
import argparse
import glob
from collections import defaultdict
from skimage.feature import register_translation
from skimage.transform import rescale
from scipy.ndimage import fourier_shift
import imageio
from ast import literal_eval

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	parser.add_argument('--blank-round', help='Round in which blank images were collected')
	parser.add_argument('--dapi-index', help='Channel index for DAPI images',type=int,default=3)
	parser.add_argument('--multiplier', help='Factors to multiply blanks before subtraction (by channel)',default='1.5,1.15,1.35')
	parser.set_defaults(save_tiles=False)
	parser.set_defaults(save_stitched=False)
	args,_ = parser.parse_known_args()
	factors = np.array(list(literal_eval(args.multiplier)) + [1])
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		FP_blanks = glob.glob(os.path.join('%s/%s/%s/stitched.tiff' % (args.basepath,tissue,args.blank_round)))
		Blanks = [imageio.imread(fp) for fp in FP_blanks]
		FP = glob.glob(os.path.join('%s/%s/round*/stitched.tiff' % (args.basepath,tissue)))
		FP = [fp for fp in FP if fp not in FP_blanks]
		Images = [imageio.imread(fp) for fp in FP]
		min_shape = (min(im.shape[0] for im in Images+Blanks), min(im.shape[1] for im in Images+Blanks))
		Images = [im[:min_shape[0],:min_shape[1]] for im in Images]
		Blanks = [im[:min_shape[0],:min_shape[1]] for im in Blanks]
		Images1 = [im[:,:,args.dapi_index] for im in Images]
		Blanks1 = [im[:,:,args.dapi_index] for im in Blanks]
		shifts = [find_shift(im,Blanks1[0]) for im in Images1]
		max_value = np.iinfo(Images[0].dtype).max
		Blanks = Blanks[0]
		# setting the 1% brightest spots to the max, so that they will certainly be set to zero
		thresh = np.percentile(Blanks,99,axis=0)
		Blanks[Blanks > thresh] = max_value
		for i,im in enumerate(Images):
			shifted_blanks = apply_shifts(Blanks,shifts[i])
			im_new = im - shifted_blanks*factors
			im_new[im_new < 0] = 0
			im_new = np.rint(im_new).astype(im.dtype)
			im_new[:,:,args.dapi_index] = im[:,:,args.dapi_index]
			imageio.imwrite(FP[i].replace('.tiff','.background_subtract.tiff'),im_new)
		print(tissue)

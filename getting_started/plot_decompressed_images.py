import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import glob,os,sys
from scipy import ndimage
import imageio
from scipy.spatial import distance
from skimage.measure import compare_ssim as ssim


def get_fov_from_patches(fov,width,height,channels,tiledir='tiled'):
	FP = glob.glob(os.path.join('%s/fov_%d.*.npy' % (tiledir,fov)))
	X = np.zeros((height,width,channels))
	for i,fp in enumerate(FP):
		r,c = fp.split('.')[1].split('-')
		r = int(r)
		c = int(c)
		x = np.load(fp)
		X[r:r+x.shape[0],c:c+x.shape[1]] = x
	return X

def normalize_image(im,gain=8,upper=99.9,cutoff=0.25):
	min_upper_thresh = np.iinfo(im.dtype).max/500
	upper_thresh = max(np.percentile(im,upper),min_upper_thresh)
	im[im > upper_thresh] = upper_thresh
	im = im/upper_thresh
	im = 1/(1 + np.exp(gain*(cutoff - im)))
	return im

def make_stitched_plots(fovs,tiledir,Genes,outdir,full_width=1024,full_height=1024,filter_size=0,outtype=np.uint8,normalize_each=True):
	max_value = np.iinfo(outtype).max
	num_channels = len(Genes)
	# load the snake pattern of fov / patches
	# this is normally saved as modified_fov_pattern.npy during preprocessing
	# for this getting_started script we'll just hardcode it
	FOV_pattern = np.array([[27,28]])
	Stitch = np.zeros((full_height*FOV_pattern.shape[0],full_width*FOV_pattern.shape[1],num_channels-1))
	for fov in fovs:
		# reconstructed images
		X = get_fov_from_patches(fov,full_width,full_height,num_channels-1,tiledir=tiledir)
		fov_pos = np.where(FOV_pattern == fov)
		Stitch[full_height*fov_pos[0][0]:full_height*(fov_pos[0][0]+1), full_width*fov_pos[1][0]:full_width*(fov_pos[1][0]+1)] = X
	Stitch = Stitch/Stitch.max()*max_value
	for i in range(Stitch.shape[2]):
		x = np.rint(Stitch[:,:,i]).astype(outtype)
		if normalize_each:
			x = normalize_image(x)
			x[x > 1] = 1
			x[x < 0] = 0
			x = np.rint(x*max_value).astype(outtype)
		gene = Genes[i]
		imageio.imwrite('%s/%s.png' % (outdir,gene),x)


if __name__ == '__main__':
	# Load the list of selected genes (genes are in the same order as in the arrays we'll load)
	f = open('genes.txt')
	Genes = [line.strip() for line in f]
	Genes += ['DAPI']
	f.close()
	os.system('mkdir decompressed_images')
	make_stitched_plots([27,28],'output',Genes,'decompressed_images')


	


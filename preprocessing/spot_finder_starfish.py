import numpy as np
import imageio
import argparse
from starfish import data, FieldOfView
from starfish.types import Axes, Features
from starfish.image import Filter
from starfish.types import Clip
from starfish.core.imagestack.imagestack import ImageStack
from starfish.spots import DecodeSpots, FindSpots
from starfish import Codebook


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Path to image (single round/channel)')
	parser.add_argument('--out-path', help='Path to save output')
	parser.add_argument('--intensity-ptile', help='Percentile threshold for pixel intensity',type=float,default=99.995)
	parser.add_argument('--filter-width', help='High pass filter width (pixels)',type=int,default=2)
	parser.add_argument('--small-peak-min', help='Min area of small peaks',type=int,default=4)
	parser.add_argument('--small-peak-max', help='Max area of small peaks',type=int,default=100)
	parser.add_argument('--big-peak-min', help='Min area of big peaks',type=int,default=25)
	parser.add_argument('--big-peak-max', help='Max area of big peaks',type=int,default=10000)
	parser.add_argument('--small-peak-dist', help='Min distance between small peaks',type=float,default=2)
	parser.add_argument('--big-peak-dist', help='Min distance between big peaks',type=float,default=0.75)
	parser.add_argument('--block-dim-fraction', help='Size of blocks to process as separated chunks (as fraction of image dim)',type=float,default=0.5)
	parser.add_argument('--spot-pad-pixels', help='Number of pixels to add in each direction around each spot',type=int,default=2)
	parser.add_argument('--existing-flat-field', help='Path to existing flat field',default=None)
	parser.add_argument('--keep-existing', help='Keep existing stitched files (if present)',dest='keep_existing', action='store_true')
	parser.set_defaults(keep_existing=False)
	args,_ = parser.parse_known_args()
	imarr = imageio.volread(args.path)
	if 'DAPI' in args.out_path.split('/')[-1]:
		imageio.imsave(args.out_path,imarr.max(0))
	else:
		imarr_orig = np.copy(imarr)
		#adds a "channel" dimension
		imarr = np.expand_dims(imarr,axis=0)
		#adds a "round" dimension
		imarr = np.expand_dims(imarr,axis=0)
		thresh = np.percentile(imarr,args.intensity_ptile)
		imarr[imarr > thresh] = thresh + (np.log(imarr[imarr > thresh] - thresh)/np.log(1.1)).astype(imarr.dtype)
		bandpass = Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
		lmp_small = FindSpots.LocalMaxPeakFinder(
			min_distance=args.small_peak_dist,
			stringency=0,
			min_obj_area=args.small_peak_min,
			max_obj_area=args.small_peak_max,
			min_num_spots_detected=2500,
			is_volume=False,
			verbose=False
		)
		lmp_big = FindSpots.LocalMaxPeakFinder(
			min_distance=args.big_peak_dist,
			stringency=0,
			min_obj_area=args.big_peak_min,
			max_obj_area=args.big_peak_max,
			min_num_spots_detected=2500,
			is_volume=False,
			verbose=False
		)
		sd = Codebook.synthetic_one_hot_codebook(n_round=1, n_channel=1, n_codes=1)
		decoder = DecodeSpots.PerRoundMaxChannel(codebook=sd)
		block_dim = int(max(imarr.shape)*args.block_dim_fraction)
		SpotCoords = np.zeros((0,2),dtype=np.int64)
		for i in range(0,imarr.shape[-2]-1,block_dim): # subtracting 1 from range because starfish breaks with x or y axis size of 1
			for j in range(0,imarr.shape[-1]-1,block_dim):
				imgs = ImageStack.from_numpy(imarr[:,:,:,i:i+block_dim,j:j+block_dim])
				imgs = bandpass.run(imgs).reduce({Axes.ZPLANE}, func="max")
				spots = lmp_small.run(imgs)
				decoded_intensities = decoder.run(spots=spots)
				spot_coords_small = np.stack([decoded_intensities[Axes.Y.value], decoded_intensities[Axes.X.value]]).T
				spots = lmp_big.run(imgs)
				decoded_intensities = decoder.run(spots=spots)
				spot_coords_big = np.stack([decoded_intensities[Axes.Y.value], decoded_intensities[Axes.X.value]]).T
				spot_coords = np.vstack([spot_coords_small,spot_coords_big])
				spot_coords[:,0] += i
				spot_coords[:,1] += j
				SpotCoords = np.vstack([SpotCoords,spot_coords])
		img_spots = np.zeros(imarr.shape[-2:],dtype=np.uint16)
		for spot in SpotCoords:
			for i in range(max(0,spot[0]-args.spot_pad_pixels),min(spot[0]+args.spot_pad_pixels+1,img_spots.shape[0])):
				for j in range(max(0,spot[1]-args.spot_pad_pixels),min(spot[1]+args.spot_pad_pixels+1,img_spots.shape[1])):
					img_spots[i,j] = imarr_orig[:,i,j].max(0)
		imageio.imsave(args.out_path,img_spots)



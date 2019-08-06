import numpy as np
from scipy import ndimage
import imageio
from nd2reader import ND2Reader
import glob,os
from collections import defaultdict
import argparse

def parse_rounds(fp):
	Rounds2Channels = defaultdict(list)
	f = open(fp,encoding='utf-8')
	header = f.readline()
	for line in f:
		ls = line.strip().split(',')
		Rounds2Channels[(ls[0],ls[1])].append(ls[2])
	f.close()
	return Rounds2Channels

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputdir', help='Path to input images')
	parser.add_argument('--images-to-process', help='CSV with nd2 files, tissue, round')
	parser.add_argument('--rounds-to-channels', help='CSV list of tissue, round, composition labels')
	parser.add_argument('--outdir', help='Output path')
	parser.add_argument('--jobnum', help='Job index for running on cluster',type=int,default=-1)
	args,_ = parser.parse_known_args()
	Rounds2Channels = parse_rounds(args.rounds_to_channels)
	f = open(args.images_to_process)
	FP = [line.strip().split(',') for line in f]
	f.close()
	FP = [('%s/%s' % (args.inputdir,fp[0]), fp[1], fp[2]) for fp in FP]
	if args.jobnum > -1:
		FP = FP[args.jobnum:args.jobnum+1]
	_=os.system('mkdir %s/FilteredCompositeImages' % args.outdir)
	for fp in sorted(FP):
		tissue = fp[1]
		round = fp[2]
		_=os.system('mkdir %s/FilteredCompositeImages/%s' % (args.outdir,tissue))
		_=os.system('mkdir %s/FilteredCompositeImages/%s/%s' % (args.outdir,tissue,round))
		with ND2Reader(fp[0]) as images:
			# iterate over fields of view, then channels
			# max over the z-stack
			try:
				images.iter_axes = 'vcz'
			except:
				images.iter_axes = 'cz'
			num_z = len(images.metadata['z_levels'])
			I = []
			i = 0
			for fov_chann_z in images:
				if i%num_z == 0:
					if i > 0:
						fov_chann = np.array(Iz).max(0)
						I.append(fov_chann)
					Iz = []
				Iz.append(fov_chann_z)
				i += 1
		fov_chann = np.array(Iz).max(0)
		I.append(fov_chann)
		FOV = images.metadata['fields_of_view']
		Channels = images.metadata['channels']
		nc = len(Rounds2Channels[(tissue,round)]) + 1
		for i,f in enumerate(FOV):
			for j,c in enumerate(Channels[:nc]):
				im = I[i*len(Channels) + j]
				if c == 'conf-405':
					channel = 'DAPI'
				else:
					channel = Rounds2Channels[(tissue,round)][j]
				channel = '%s.%s' % (channel,c)
				if i == 0:
					_= os.system('mkdir %s/FilteredCompositeImages/%s/%s/%s' % (args.outdir,tissue,round,channel))
				imageio.imwrite('%s/FilteredCompositeImages/%s/%s/%s/fov_%d.png' % (args.outdir,tissue,round,channel,f),im)
		print(tissue,round)


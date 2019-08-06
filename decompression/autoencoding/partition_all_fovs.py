import argparse
import glob,os
from collections import defaultdict
from random import shuffle, choice

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--basepath', help='Path to directory with subdirs for parsed images in each tissue')
	parser.add_argument('--outpath', help='Path to directory to write lists of tfrecords')
	parser.add_argument('--num-fov', help='Number of fovs to include in one block',type=int,default=21)
	args,_ = parser.parse_known_args()
	FP = glob.glob(os.path.join(args.basepath,'tissue*','tfrecords','fov_*.tfrecords'))
	All_FOV_FP = defaultdict(list)
	Remaining_FOV_FP = defaultdict(list)
	for fp in FP:
		fov = int(fp.split('/')[-1].split('_')[1].split('.')[0])
		All_FOV_FP[fov].append(fp)
		Remaining_FOV_FP[fov].append(fp)
	for fov in All_FOV_FP:
		shuffle(All_FOV_FP[fov])
		shuffle(Remaining_FOV_FP[fov])
	max_fov = max(All_FOV_FP.keys())
	FOV_Blocks = []
	for fov in range(max_fov-args.num_fov+2):
		for fp in Remaining_FOV_FP[fov]:
			fov_block = [fp]
			for i in range(fov+1,fov+args.num_fov):
				try:
					fov_block.append(Remaining_FOV_FP[i].pop())
				except:
					fov_block.append(choice(All_FOV_FP[i]))
			assert len(fov_block) == args.num_fov
			FOV_Blocks.append(fov_block)
		Remaining_FOV_FP[fov] = []
	for fov in range(max_fov,args.num_fov-1,-1):
		for fp in Remaining_FOV_FP[fov]:
			fov_block = [fp]
			for i in range(fov-1,fov-args.num_fov,-1):
				try:
					fov_block.append(Remaining_FOV_FP[i].pop())
				except:
					fov_block.append(choice(All_FOV_FP[i]))
			assert len(fov_block) == args.num_fov
			FOV_Blocks.append(fov_block)
	for i,b in enumerate(FOV_Blocks):
		f = open('%s/tfrecords_files.%d.txt' % (args.outpath,i),'w')
		f.write(' '.join(b))
		f.close()
	
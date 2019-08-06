import numpy as np
import tensorflow as tf
import os
import glob
import argparse
from scipy.sparse import load_npz, csr_matrix, diags
import imageio

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	if isinstance(value,list):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
	else:
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_cell_masks_in_patch(CellMasks,fov_mask,patch_pos,patch_height,patch_width,min_pixels=512):
	patch_mask = np.zeros(fov_mask.shape,dtype=np.int)
	patch_mask[patch_pos[0]:patch_pos[0]+patch_height, patch_pos[1]:patch_pos[1]+patch_width] = 1
	pm = diags(patch_mask.flatten(),0)
	cell_patch_mask = CellMasks.dot(pm)
	# remove cells with few pixels
	cell_patch_mask = cell_patch_mask[cell_patch_mask.getnnz(1) > min_pixels]
	# adjust indices from full image to within fov
	col_idx = cell_patch_mask.indices % fov_mask.shape[1]
	row_idx = (cell_patch_mask.indices - col_idx) / fov_mask.shape[1]
	col_idx = col_idx - patch_pos[1]
	row_idx = row_idx - patch_pos[0]
	flat_idx = (row_idx*patch_width + col_idx).astype(np.int64)
	n_cells = len(cell_patch_mask.indptr)-1
	sp_embedding_mask_row = []
	sp_embedding_mask_col = []
	sp_embedding_mask_values = []
	for i in range(n_cells):
		for j in range(cell_patch_mask.indptr[i],cell_patch_mask.indptr[i+1]):
			sp_embedding_mask_row.append(i)
			sp_embedding_mask_col.append(j-cell_patch_mask.indptr[i])
			sp_embedding_mask_values.append(flat_idx[j])
	if len(sp_embedding_mask_col) > 0:
		max_pixels_per_cell = max(sp_embedding_mask_col) + 1
	else:
		max_pixels_per_cell = 0
	return n_cells, max_pixels_per_cell, sp_embedding_mask_row, sp_embedding_mask_col, sp_embedding_mask_values

def write_record_from_numpy(image,validation,validation_indices,tissue,fov,fov_mask,CellMasks,tfrecords_filename,patch_height=256,patch_width=256):
	tissue_byte = tf.compat.as_bytes(tissue)
	writer = tf.python_io.TFRecordWriter(tfrecords_filename + '.tfrecords')
	fov_pos = (np.where(fov_mask.sum(1))[0].min(), np.where(fov_mask.sum(0))[0].min())
	for i in range(0,image.shape[0],patch_height):
		for j in range(0,image.shape[1],patch_width):
			image_raw = image[i:i+patch_height,j:j+patch_width]
			# Don't write records for patches with nothing in them
			if image_raw.sum() > 0:
				patch_pos = (i + fov_pos[0], j + fov_pos[1])
				sp_embedding = get_cell_masks_in_patch(CellMasks,fov_mask,patch_pos,patch_height,patch_width)
				validation_raw = validation[i:i+patch_height,j:j+patch_width]
				h,w,c = image_raw.shape
				image_raw = image_raw.tostring()
				validation_raw = validation_raw.tostring()
				example = tf.train.Example(features=tf.train.Features(feature={'raw_data': _bytes_feature(image_raw),
				'height': _int64_feature(h),
				'width': _int64_feature(w),
				'channels': _int64_feature(c),
				'tissue': _bytes_feature(tissue_byte),
				'fov': _int64_feature(fov),
				'row_offset': _int64_feature(i),
				'col_offset': _int64_feature(j),
				'raw_validation': _bytes_feature(validation_raw),
				'validation_indices': _int64_feature(validation_indices),
				'n_cells': _int64_feature(sp_embedding[0]),
				'max_pixels_per_cell': _int64_feature(sp_embedding[1]),
				'sp_embedding_mask_row': _int64_feature(sp_embedding[2]),
				'sp_embedding_mask_col': _int64_feature(sp_embedding[3]),
				'sp_embedding_mask_values': _int64_feature(sp_embedding[4])}))
				writer.write(example.SerializeToString())
	writer.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--patch-width', help='Patch width',type=int,default=256)
	parser.add_argument('--patch-height', help='Patch height',type=int,default=256)
	parser.add_argument('--basepath', help='Path to filtered composite images')
	parser.add_argument('--genes', help='Path to gene labels')
	parser.add_argument('--tissues', help='Comma-separated list of tissue numbers to include')
	args,_ = parser.parse_known_args()
	f = open(args.genes)
	Genes = [line.strip() for line in f]
	f.close()
	for t in args.tissues.split(','):
		tissue = 'tissue%s' % t
		os.system('mkdir %s/%s/tfrecords' % (args.basepath,tissue))
		CellMasks = load_npz('%s/%s/segmented/cell_masks.npz' % (args.basepath,tissue))
		full_size = imageio.imread('%s/%s/stitched_aligned_filtered/DAPI.tiff' % (args.basepath,tissue)).shape
		FP = glob.glob(os.path.join('%s/%s/arrays_aligned_filtered/fov_*.Composite_0.npy' % (args.basepath,tissue)))
		fovs = [fp.split('/')[-1].split('.')[0].split('_')[-1] for fp in FP]
		fov_pattern = np.load('%s/%s/modified_fov_pattern.npy' % (args.basepath,tissue))
		for fov in fovs:
			FP = glob.glob(os.path.join('%s/%s/arrays_aligned_filtered/fov_%s.*' % (args.basepath,tissue,fov)))
			FP_composite = [fp for fp in FP if 'Composite' in fp.split('/')[-1]]
			x = np.load(FP_composite[0])
			width,height = x.shape
			Im_FOV = np.zeros((width,height,len(FP_composite)),dtype=np.float32)
			for fp in FP_composite:
				channel_num = int(fp.split('/')[-1].split('.')[1].split('_')[1])
				x = np.load(fp).astype(np.float32)
				Im_FOV[:,:,channel_num] = x
			FP_validation = [fp for fp in FP if ('Composite' not in fp.split('/')[-1]) and ('DAPI' not in fp)]
			FP_validation = [fp for fp in FP_validation if fp.split('/')[-1].split('.')[1] in Genes]
			Im_FOV_validation = np.zeros((width,height,len(FP_validation)),dtype=np.float32)
			indices = []
			for i,fp in enumerate(FP_validation):
				channel_num = Genes.index(fp.split('/')[-1].split('.')[1])
				x = np.load(fp).astype(np.float32)
				Im_FOV_validation[:,:,i] = x
				indices.append(channel_num)
			fov_mask = np.zeros((width*fov_pattern.shape[0],height*fov_pattern.shape[1]),dtype=np.bool)
			fov_idx = np.where(fov_pattern == int(fov))
			fov_mask[fov_idx[0][0]*width:(fov_idx[0][0] + 1)*width, fov_idx[1][0]*height:(fov_idx[1][0] + 1)*height] = True
			fov_mask = fov_mask[:full_size[0],:full_size[1]]
			write_record_from_numpy(Im_FOV,Im_FOV_validation,indices,tissue,int(fov),fov_mask,CellMasks,'%s/%s/tfrecords/fov_%s' % (args.basepath,tissue,fov), patch_width=args.patch_width, patch_height=args.patch_height)
			print(tissue,fov)

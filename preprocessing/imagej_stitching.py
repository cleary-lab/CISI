import imagej
import argparse
import glob
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--source-paths', help='Paths to directories for each round for each tissue')
	parser.add_argument('--cols', help='Number of columns')
	parser.add_argument('--rows', help='Number of rows')
	parser.add_argument('--fijipath', help='Path to Fiji.app')
	parser.add_argument('--grid-type',help='Stitching pattern',default='snake-by-rows')
	parser.add_argument('--overlap',help='Percent overlap in tiles',type=int,default=10)
	args, _ = parser.parse_known_args()
	grid = ' '.join(args.grid_type.split('-'))
	round_directories = glob.glob(args.source_paths)

	# ImageJ/Fiji setup
	ij = imagej.init(args.fijipath)
	IJM_EXTENSION = ".ijm"
	STITCHING_MACRO = """
	#@ String sourceDirectory
	#@ String outputDirectory
	#@ int overlap
	#@ String grid
	#@ int cols
	#@ int rows

	print(outputDirectory);
	function action(sourceDirectory, outputDirectory, genericFilename, filename, coordinateFilename) {
	run("Grid/Collection stitching", "type=[Grid: " + grid + "] order=[Right & Down ] grid_size_x=" + cols + " grid_size_y=" + rows + " tile_overlap=" + overlap + " first_file_index_i=0 directory=" + sourceDirectory + " file_names=" + genericFilename + " output_textfile_name=" + coordinateFilename + " fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 subpixel_accuracy compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
	saveAs("png", outputDirectory + filename);
	close();
	}

	action(sourceDirectory, outputDirectory, "fov_{i}.png", "/reference_stitch", "TileConfiguration.txt");

	"""
	cols = args.cols
	rows = args.rows
	overlap = args.overlap
	for round_directory in round_directories:
		try:
			dapi_directory = glob.glob(os.path.join(round_directory, "DAPI*/"))[0]
			print(dapi_directory)
			args = {
			'sourceDirectory': dapi_directory,
			'outputDirectory': round_directory,
			'overlap': overlap,
			'grid': grid,
			'cols': cols,
			'rows': rows
			}

			result = ij.py.run_macro(STITCHING_MACRO, args)
		except:
			pass

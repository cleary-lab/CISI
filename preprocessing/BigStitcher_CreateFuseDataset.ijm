// read dataset path, number of tiles as commandline arguments
args = getArgument()
args = split(args, " ");
 
basePath = args[0];
if (!endsWith(basePath, File.separator))
{
    basePath = basePath + File.separator;
}
wellRound = args[1];
fusePath = args[2];
if (!endsWith(fusePath, File.separator))
{
    fusePath = fusePath + File.separator;
}

// wellRound = A3_round1
// basePath = Z:/projects/CTS/ImageAnalysis/InSitu_Transcriptomics/MotorCortex/Dec2019_HCR_MOp/A3_round1_ch6_ch7_ch2.nd2
// basePath2 = Z:/projects/CTS/ImageAnalysis/InSitu_Transcriptomics/MotorCortex/Dec2019_HCR_MOp
// exportPath = Z:/projects/CTS/ImageAnalysis/InSitu_Transcriptomics/MotorCortex/Dec2019_HCR_MOp/A3_round1
// outputPath = Z:/projects/CTS/ImageAnalysis/InSitu_Transcriptomics/MotorCortex/Dec2019_HCR_MOp
 
// define dataset
run("Define dataset ...",
	"define_dataset=[Automatic Loader (Bioformats based)]" +
	" project_filename=" + wellRound + ".xml path=" + basePath + wellRound + ".nd2 exclude=10" +
	" bioformats_series_are?=Tiles move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)]" +
	" how_to_load_images=[Re-save as multiresolution HDF5] dataset_save_path=" + basePath +
	" subsampling_factors=[{ {1,1,1}, {2,2,1} }]" +
	" hdf5_chunk_sizes=[{ {32,32,4}, {16,16,16} }]" +
	" timepoints_per_partition=1" +
	" setups_per_partition=0" +
	" use_deflate_compression" +
	" export_path=" + basePath + wellRound);

// calculate pairwise shifts
run("Calculate pairwise shifts ...",
	"select=" + basePath + wellRound + ".xml" +
	" process_angle=[All angles]" +
	" process_channel=[All channels]" +
	" process_illumination=[All illuminations]" +
	" process_tile=[All tiles]" +
	" process_timepoint=[All Timepoints]" +
	" method=[Phase Correlation]" +
	" channels=[use Channel conf-405]" +
	" downsample_in_x=2" +
	" downsample_in_y=2" +
	" downsample_in_z=1");

// filter shifts
run("Filter pairwise shifts ...",
	"select=" + basePath + wellRound + ".xml" +
	" filter_by_link_quality" +
	" min_r=0.5" +
	" max_r=1" +
	" max_shift_in_x=0" +
	" max_shift_in_y=0" +
	" max_shift_in_z=0" +
	" max_displacement=0");

// Global optimize and apply
run("Optimize globally and apply shifts ...",
	"select=" + basePath + wellRound + ".xml" +
	" process_angle=[All angles]" +
	" process_channel=[All channels]" +
	" process_illumination=[All illuminations]" +
	" process_tile=[All tiles]" +
	" process_timepoint=[All Timepoints]" +
	" relative=2.500" +
	" absolute=3.500" +
	" global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles]" +
	" fix_group_0-0");

// fuse tiles
run("Fuse dataset ...",
	"select=" + basePath + wellRound + ".xml" +
	" process_angle=[All angles]" +
	" process_channel=[All channels]" +
	" process_illumination=[All illuminations]" +
	" process_tile=[All tiles]" +
	" process_timepoint=[All Timepoints]" +
	" bounding_box=[Currently Selected Views]" +
	" downsampling=1 pixel_type=[16-bit unsigned integer]" +
	" interpolation=[Linear Interpolation]" +
	" image=[Precompute Image]" +
	" interest_points_for_non_rigid=[-= Disable Non-Rigid =-]" +
	" blend preserve_original produce=[Each timepoint & channel]" +
	" fused_image=[Save as (compressed) TIFF stacks]" +
	" output_file_directory=" + fusePath +
	" filename_addition=" + wellRound);

// quit after we are finished
eval("script", "System.exit(0);");


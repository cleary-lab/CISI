Steps taken during preprocessing of a CISI dataset:

1. Run BigStitcher_ArrayJob.sh to create h5 datasets, and fuse tiles (into tmp location)
2. Create tissue_channel_list.txt (well_round_ch1_ch2_ch3 tissue1 ch1 ch2 ch3)
3. Run SpotFinder_ArrayJob.sh --> save one file per round / channel in FilteredCompositeImages
4. Run save_registered_images.py to register across rounds
		e.g. $ python save_registered_images.py --basepath ../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/ --rounds-to-channels ../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/rounds2channels.txt --processed-filename DAPI.tiff --rescale-intensity --save-stitched --tissues 5
		In some cases the BigStitcher fusion may have gone funky in certain regions of the tissue in one or more rounds, making global alignment difficult. When this happens we try to align locally, in blocks, using the --shift-blocks option:
		$ python save_registered_images.py --basepath ../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/ --rounds-to-channels ../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/rounds2channels.txt --processed-filename DAPI.tiff --rescale-intensity --save-stitched --shift-blocks --tissues 7
5. Run merge_all_composites.py to generate total RNA image for each tissue
6. Segment cells with CellProfiler, saving one image mask per object in each tissue. Steps taken with 20x objective:
		Load image (DAPI and All_Composite)
		Names and types (DAPI and RNA)
		IdentifyPrimaryObject (14,45) (Intensity, Intensity)
		IdentifySecondaryObject (RNA from All_Composite)
		ConvertObjectsToImage (uint16) --> pixels for each object have a separate value
		SaveImage (32 bit)
7. Run segmentation_sparse_matrix to generate a sparse matrix with cell masks (on flattened image array)
8. Run segmented_integrated_intensity
9. Use Fiji to create masks (Mask.tif in stitched_aligned_filtered) that exclude folded regions
		Look at all pairs of (round1, round_i) and mask regions that failed to align, had torn / folded tissue, etc
9. (optional -- needed only for autoencoding method) Run write_tf_records
		e.g. $ python write_tfrecords.py --basepath ../../MotorCortex/20200201_MotorCortex_HCR/FilteredCompositeImages/ --genes ../../MotorCortex/Training/Simulations_38genes/genes.txt --tissues 8 --filter-size 6

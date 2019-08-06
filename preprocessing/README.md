Steps taken during preprocessing:

1. run ParseND2_ArrayJob.sh, which runs parse_and_z_project for each nd2 file
2. run imagej_stitching to stitch DAPI channel for every tissue/round
3. run StitchAndFilter_ArrayJob, which runs stitch_from_fiji_coords to stitch and apply median filter
4. run subtract_background
5. use Fiji to determine intensity thresholds for each channel (setting min to 0, max to max from auto adjust)
6. run apply_channel_thresholds
7. (optional) run calculate_and_apply_flat_field_correction
8. run save_registered_images to register across rounds and break into tiles
9. segment cells with CellProfiler, saving one image mask per object in each tissue
10. run segmentation_sparse_matrix to generate a sparse matrix with cell masks (on flattened image array)
11. run write_tf_records
12. run segmented_integrated_intensity

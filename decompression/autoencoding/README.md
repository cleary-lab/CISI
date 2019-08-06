CISI-AutoEncoding

The methods here are meant to be run after training (to learn the dictionary) and image preprocessing

Run a test job

1. Create a data/ directory. Copy the dictionary (gene_modules.npy), composition matrix (phi.npy), gene-gene correlations (correlations.py), and relative abundance estimates (relative_abundance.npy) to this directory. (These are all files created during training.)

2. Select a subset of individual fields of view (fovs) for hyperparameter testing. The fovs can come from different sections, but should collectively be unique and contiguous (i.e., don't include the same fov from different sections, and ensure there are no gaps in the final range of fovs). Copy the .tfrecords for these fovs to data/

3. Run training with the following input (in addition to any hyperparameters you wish to specify; also, make sure --fovs argument matches the range chosen in Step 2):
	$ python module/run_task.py \
	--job-dir trained_model/ \
	--train-files data/ \
	--eval-files data/ \
	--fovs 3-12 \
	--train-steps 200 \
	--decompress-steps 1500 \
	--num-layers 4 \
	--loss-fn l1

4. Use the trained model to run predictions (outputting numpy arrays of size patch width x patch height x num_genes for each patch in each fov)
	$ python module/run_task.py \
	--job-dir trained_model/ \
	--train-files data/ \
	--eval-files data/ \
	--fovs 3-12 \
	--num-layers 4 \
	--loss-fn l1 \
	--predict-only \
	--predict-dir decompressed_data/


Hyperparameter Testing

1. (as above for running a test job)

2. Select a subset of individual fields of view (fovs) for hyperparameter testing. For example, select 3 fovs from 3 different tissue sections (eg fovs 3,6,9 & 4,7,10 & 5,8,11 from 3 tissue sections). The fovs from different sections should collectively be unique and contiguous (i.e., don't include the same fov from different sections, and ensure there are no gaps in the final set of fovs, as in the example given). Copy the .tfrecords for these fovs to data/

3. Use Google Cloud hyperparameter tuning to test the range of hyperparameters defined in hptuning_config.yaml. Additional parameters (e.g. --num-layers) could be included in the config file (although for we tend to run different values of --num-layers as separate hptuning jobs). Set the fov range to match the range of fovs selected in Step 2 (python-style range, so 3-12 in the example above). Run the steps in ml-engine.sh

4. The tuning algorithm will adaptively search for parameters that maximize the correlation between direct measurements and recovered signal within segmentation masks (this correlation is used for evaluation, but not for training the model).


Training the model on all available data with fixed hyperparameters

1. Set a PROJECT_DIR env variable to point to the directory housing FilteredCompositeImages, ReconstructedImages, etc., and run:
	$ python partition_all_fovs.py \
	--basepath $PROJECT_DIR/FilteredCompositeImages \
	--outpath $PROJECT_DIR/FilteredCompositeImages/ShuffledTFRecordBlocks \
	--num-fov 12
	
	The argument to --num-fov should match the number of fovs selected during tuning (9 in the example above).
	The output of this will be a series of files, each containing a list of contiguous fovs from different tissues, such that all fovs from all tissues are represented across the lists. One training job will be run, and one model will be trained per list. We find that training models on data from multiple sections in this manner helps with stability, especially in cases when some tissue sections only contain a subset of genes expressed.

2. Set the optimized values from hyperparameter tuning to the appropriate arguments in ml-engine.final_models.sh

3. Create a data/ directory. Copy the dictionary (gene_modules.npy), composition matrix (phi.npy), gene-gene correlations (correlations.py), and relative abundance estimates (relative_abundance.npy) to this directory.

4. Run ml-engine.final_models.sh to train models on all blocks of data (Make sure START and END match the range of ShuffledTFRecordBlocks/tfrecords_files.*.txt)

5. Set a $MODEL_DATE env variable to match the date in trained model output directories. Run cp_models.sh

6. Set the optimized values from hyperparameter tuning to the appropriate arguments in run_predictions.sh (most important to set --num-layers and --num-filters). Run run_predictions.sh (note that fields selected for 'tissue' and 'fov' might need adjusting in lines 40 & 41)

This will output numpy arrays of size (patch width x patch height x num_genes) for each patch in each fov.



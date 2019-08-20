# Getting Started

Here we'll run autoencoding-based decompression on a small amount of data used in our study. We'll start with pre-processed data (in the form of tfrecords), train an autoencoder on the 10 composite signals, learn a model to estimate the encoded representation of each unobserved gene, then produce decompressed images for all 37 individual genes.

### Dependencies

This code has been tested with python3.7 and tensorflow 1.14
We use Anaconda to set up an appropriate environment (this may take a few minutes):

    conda create -n cisi_test tensorflow tensorflow-probability matplotlib skimage
    conda activate cisi_test


In some cases we've found that a few dependencies for tensorflow-probability are not installed. Run the following to check, and install any extras (using conda) if necessary.

    python -c 'import tensorflow_probability'

### Run the code

The fovs used here are real data from our original study (fovs 27 and 28 from tissue 7), and correspond to an area surrounding the highlighted region from Supplemental Figure 4 (tissue 7). The input data are stored as tfrecords in 'data/'. We also include images of each of the composite signals in 'original_images/'.

The input to training also include a few additional matrices:
	phi.npy is the composition matrix, defining the genes included in each measurement
	gene_modules.npy is the gene module matrix, learned from snRNA-Seq data (see README in training/)
	correlations.npy is a gene-gene correlation matrix (also from snRNA)
	relative_abundance.npy is an estimate of the relative abundance of each gene (also from snRNA)

We include the autoencoder parameters that we used in the paper (as learned from hyperparameter testing; see README in decompression/autoencoding) in train_model.sh. Feel free to play with these to see how the results change.

To run the analysis, first train the autoencoder on a couple fields of view and save the output of decompression as numpy arrays (this may take a few minutes).

    bash train_model.sh

This will create a few directories. 'model/' contains the trained model. 'output/' contains the output of decompression (one multi-dimensional array per small patch). 'output_encoded/' similarly contains the encoded representation of each gene.

Next, plot the results:

    python plot_decompressed_images.py

This should produce 37 images, one per gene, saved in 'decompressed_images/'.

### Comparison with validation images

For validation purposes, we directly image a subset of genes in each tissue section (in addition to the composites). In this section we measured Flt1, Gad1, Vip, Vtn, and Xdh. You can find the validation images for these genes (in green), merged together with results of decompression presented in the paper (in magenta) in 'original_images/'. Since our original results are from a model trained on a much larger dataset (and trained for many more iterations), you may find that the decompressed results from this small dataset differ slightly.

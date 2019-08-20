# CISI
Composite In Situ Imaging (CISI) is a method developed to increase throughput (multiplexing over genes, and larger tissue volumes) in imaging transcriptomics. In a CISI experiment, we generate a series of composite images -- corresponding to linear combinations of genes, and made by simultaneous probing for multiple targets -- which are decompressed based on the mathematics of compressed sensing.

# Getting started

Run a [demo](getting_started/) of the autoencoding-based decompression on composite data from our study.

# Computational overview of the full analysis

1. Follow the training steps to analyze snRNA-Seq data. This will generate a gene module dictionary and use simulation to select the best measurement compositions.

(Not included in this repository: one will then order and mix probes according to the selected composite designs, and generate a series of composite images, potentially in multiple tissue sections.)

2. Preprocess the imaging data to generate background subtracted, smoothed, stitched images, and cell segmentation masks.

3. Decompress the data using either the autoencoding-based method, or the method based on segmentation.
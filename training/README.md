Prior to decompression, we analyze snRNA-Seq (or scRNA-Seq) data and simulate compressed sensing to find a module dictionary and determine appropriate numbers and compositions of measurements.

Steps to run simulations:

1. Make a data/ directory and place your expression data there (data.tpm.npy in the example below). Matrix should be size (genes x cells). Place a numpy array with all gene labels in the same directory (labels.all_genes.npy below). Finally, create a txt file with a list of the genes (one per line) selected for profiling by CISI (selected_genes.txt).

2. Run the simulations (this will take some time):

	$ python find_modules.py \
	--outpath Simulations
	--all-genes data/labels.all_genes.npy \
	--select-genes data/selected_genes.txt \
	--datapath data/data.tpm.npy \
	--dict-size 80 \
	--k-sparsity 3 \
	--lasso-sparsity 0.2 \
	--num-measurements 10 \
	--max-compositions 3 \
	--mode G

3. For given arguments to --num-measurements and --max-compositions, results will be saved to Simulations/*_measurements/*_max. The gene module dictionary will be saved there as gene_modules.npy. A summary of the results with the best compositions will be saved in simulation_results.txt, and the actual compositions corresponding to each result included in the summary will be saved in *_measurements/*_max/measurement_compositions/version_*.txt. Typically, we select the first result, corresponding to the composition with the highest overall pearson correlation.

4. For the selected result (e.g., "version 12" in simulation_results.txt), format the compositions (from measurement_compositions/version_12.txt) into a csv with header ("Channel index,Gene"). On each line, write one pair of channel index (i.e., measurement number, indexed to 0) and gene. (Note that we keep this step somewhat manual, to accommodate any adjusting of compositions due to availability of probes, etc.)

5. Generate a numpy array of the measurement compositions:

	$ python make_phi.py \
	--composites Simulations/10_measurements/3_max/compositions.csv \
	--genes data/selected_genes.txt \
	--savepath Simulations/10_measurements/3_max


6. Calculate the relative abundance of each gene:

	$ python make_prior_abundance.py \
	--all-genes data/labels.all_genes.npy \
	--selected-genes data/selected_genes.txt \
	--datapath data/data.tpm.npy \
	--savepath data/


7. Calculate correlations between all selected genes:

	$ python make_correlations.py \
	--datapath data/data.selected_genes.npy \
	--savepath data/


8. Calculate conditional probabilities of expression:

	$ python make_conditional_probability.py \
	--datapath data/data.selected_genes.npy \
	--savepath data/

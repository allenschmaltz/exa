# Replication scripts for the additional results in the Online Appendix

For the additional results in the Online Appendix, we consider the FCE data, the BEA 2019 data, and the CoNLL 2019 data. The FCE data is the same as that in the main text, and the latter two datasets are processed via the instructions below. In addition to training and evaluating our models, we also provide notes on running the baselines of the previous works appearing in the Online Appendix. We include instructions for each of the datasets in turn below. The core dependencies are the same as with the experiments in the main text, as noted in [README_dependencies.md](README_dependencies.md).

Note that these bash scripts are grouped into a minimal number of files for organizational simplicity, and contain code blocks intended to be run separately rather than calling the script directly from the command line. NOTE, in particular, that after training, the applicable epoch number will need to be updated with the variable `EPOCH=X` for inference to choose the appropriate model file, which may differ from the hard-coded values in the provided examples.

For reference to check that the output is as expected (and as an aid to map a particular code block to the applicable table/figure/section/etc. of the paper), in some cases we include expected output in comments after the code block. As with those for the experiments of the main text, these scripts can be viewed as a combination of replication notes for the presented experiments, and tutorials/examples to guide how to run these methods on additional datasets.

## FCE data -- Additional Results

We re-run the same model as in the main text, but with BERT_base for comparison. The FCE data preprocessing is the same as that for the main experiments. Additionally, we run a model with only 2 filters (M=2) for reference, as described in the Online Appendix.

Train the base sentence-level model (with BERT_base) and evaluate at the token-level using the proposed approach: [fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel_using_bertbase.sh](fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel_using_bertbase.sh)

Train a 2-filter (M=2) version of the base sentence-level model (with BERT_base) and evaluate at the token-level using the proposed approach: [fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel_using_bertbase_only2filters.sh](fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel_using_bertbase_only2filters.sh)

## BEA 2019 Data Experiments

First, we preprocess the BEA 2019 data as in Bujel, Yannakoudakis, and Rei (2021), using the specified indexes of the training set as the dev set, and the original Shared Task test set as the held-out eval set. The data is available for download from [https://www.cl.cam.ac.uk/research/nl/bea2019st/](https://www.cl.cam.ac.uk/research/nl/bea2019st/) and the specified indexes are available at [https://github.com/bujol12/bert-seq-interpretability/blob/master/dev_indices_train_ABC.txt](https://github.com/bujol12/bert-seq-interpretability/blob/master/dev_indices_train_ABC.txt). Preprocessing: [additional_data/bea2019/bea2019_data_init.sh](additional_data/bea2019/bea2019_data_init.sh)

Next, we train and eval both the uniCNN+BERT_base and the uniCNN+BERT_base+mm models: [additional_data/bea2019/bea2019_base_model__train_sentlevel.sh](additional_data/bea2019/bea2019_base_model__train_sentlevel.sh)

## CoNLL 2010 Data Experiments

This set of scripts is more involved, as we are going to run all of the core variations, including the K-NN, as in the experiments for the main text.

First, we process the Szeged Uncertainty Corpus from the link provided in the Online Appendix. We also convert the data to the format used by the previous models, as we re-run them with this new publicly available split for reference. Preprocessing: [additional_data/conll2010uncertainty/conll2010uncertainty_data_init.sh](additional_data/conll2010uncertainty/conll2010uncertainty_data_init.sh)

Next, we train and eval both the uniCNN+BERT_{base_uncased} and the uniCNN+BERT_{base_uncased}+mm models: [additional_data/conll2010uncertainty/zero_shot/conll2010uncertainty_base_model__train_sentlevel_uncased.sh](additional_data/conll2010uncertainty/zero_shot/conll2010uncertainty_base_model__train_sentlevel_uncased.sh)

Fine-tune for supervised labeling, uniCNN+BERT_{base_uncased}+S*: [additional_data/conll2010uncertainty/supervised/conll2010uncertainty_supervised_model__train_sentlevel_uncased.sh](additional_data/conll2010uncertainty/supervised/conll2010uncertainty_supervised_model__train_sentlevel_uncased.sh)

### Construct and evaluate the K-NN approximation (CoNLL 2010):

Generate the exemplar vectors and then cache the distances between the query and the support set for the uniCNN+BERT_{base_uncased}+mm model: [additional_data/conll2010uncertainty/knn/conll2010uncertainty_mm_model_uncased__cache_exemplar_vectors_to_disk.sh](additional_data/conll2010uncertainty/knn/conll2010uncertainty_mm_model_uncased__cache_exemplar_vectors_to_disk.sh)

Train and evaluate the distance-weighted K-NN model, and additionally evaluate using the ExAG decision rule: [additional_data/conll2010uncertainty/knn/conll2010uncertainty_mm_model__linear_exa_train_eval.sh](additional_data/conll2010uncertainty/knn/conll2010uncertainty_mm_model__linear_exa_train_eval.sh)

### Baselines from previous works (CoNLL 2010):

Instructions for running the code from Rei and Søgaard (2018): [additional_data/conll2010uncertainty/baselines_from_external_models/lstm/conll2010uncertainty_mltagger_baseline.sh](additional_data/conll2010uncertainty/baselines_from_external_models/lstm/conll2010uncertainty_mltagger_baseline.sh)

Instructions for running the code from Bujel, Yannakoudakis, and Rei (2021): [additional_data/conll2010uncertainty/baselines_from_external_models/transformer/conll2010uncertainty_transformer_baseline.sh](additional_data/conll2010uncertainty/baselines_from_external_models/transformer/conll2010uncertainty_transformer_baseline.sh)

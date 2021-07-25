# Replication scripts

First, note the dependencies in [README_dependencies.md](README_dependencies.md).

## General Notes

As a first pass, it may be easier to review the scripts for the additional results in the Online Appendix that appear in [README_replication_appendix.md](README_replication_appendix.md), since it is a smaller set of experiments, the scripts have additional in-line comments, and the file path/directory naming conventions are perhaps easier to follow given the conventions of the current version of the code. We assume below that the reader has reviewed those scripts, but in some cases, we duplicate the applicable usage instructions in these scripts, as well. (Note that we include progressively fewer in-line comments and notes for the scripts here for runs that are otherwise similar but differ in the model and/or dataset; we assume these instructions are read top to bottom. When in doubt, look at the corresponding script for an earlier model/dataset for additional instructions.)

Note that the bash scripts will require you to update the paths marked `UPDATE_WITH_YOUR_PATH`. Note that we have placed multiple runs into the bash scripts for organizational purposes, but typically, you will want to run each of the sections separately and examine the output and results, rather than just changing the paths and letting the whole script run, because some steps require choosing particular epochs (`EPOCH=X`) which may differ slightly across runs due to non-determinism. Also, note that the actual directory names and file names that we have used can be changed provided they are consistent; we have largely retained our original directories for ease of consistency in producing these scripts and to reference the runs on our own system. (Note that the `FORWARD_TYPE` variable is no longer used in the code, but is retained in some of the scripts to organize the directories of some of the original models from 2019.) However, one important point: In some cases when running, we made use of a temporary scratch directory; as such, the `SERVER_DRIVE_PATH_PREFIX` and `SERVER_SCRATCH_DRIVE_PATH_PREFIX` might appear inconsistent across some scripts, but you can just set these to be the same directory.  

In some cases, we have included key output in-line as comments so that you can quickly see if your results are similar, and as aid for verifying the correspondence to the applicable section in the paper.

Next, we provide a high-level overview of the scripts. We arrange the scripts by dataset, and by the preprocessing and train/eval of the various models.

First, we consider the experiments on the FCE grammar datasets. We have 3 main models:

1. A model trained at the sentence-level: uniCNN+BERT. This model only has access to sentence-level labels. We also demonstrate calculating aggregate feature scores with this base model. (We can do this with the other models, as well, but that functionality is not currently implemented in this release for the other models.)
1. A model initialized with the parameters of uniCNN+BERT and then fine-tuned using the min-max loss: uniCNN+BERT+mm. This model only has access to sentence-level labels.
1. A model initialized with the parameters of uniCNN+BERT and then fine-tuned using token-level labels: uniCNN+BERT+S*. This model has access to token-level labels and is a fully-supervised sequence-labeling model.

Now, with each of these three models, we are going to train and then evaluate the sequence-labeling effectiveness. Then, we build, train, and evaluate the K-NN models from the derived token-specific features, and we also produce results for the inference-time decision rules. Note that the exemplar auditing rules do not in themselves necessitate training the K-NN, but to keep things simple, in this version, all such analyses (and also the visualization of the exemplars) go through the K-NN data structures. The pipeline for analyzing the non-parametric methods is as follows:
- First, save the exemplar vectors to disk by performing a single forward pass over the data (training and eval data). In the simplest case, the training exemplars become the support set.
- Next, cache the exemplar distances to speed-up training the K-NNs. This involves matching, via exact search, each token in an eval instance to one or more token(s) in the support set. These scripts will also split the original dev set into a K-NN training set and a K-NN dev set for eval, where again as noted above, in the simplest case, the original training set is the support set.
- Then, train and evaluate the K-NNs, including on the held-out test set (which in all cases, is the original test set for the given dataset).
- Analyze the K-NN output and calculate the results using the inference-time decision rules, as applicable.

Finally, for some of the models, we repeat this process with experiments with additional domain-shifted/out-of-domain data, in some cases updating the support set.

For the FCE data, we have separated each of the three main models into individual sections below. The general pipeline is the same. The differences in particular, to note, are the following:

- uniCNN+BERT+MM and uniCNN+BERT+S* can only be trained after uniCNN+BERT, since they are initialized with its parameters.
- Note that uniCNN+BERT and uniCNN+BERT+MM should not have access to token-level labels for training or tuning for the pure zero-shot sequence labeling setting. This includes for the K-NN. Command line flags handle this behavior; see the provided bash scripts for use. In the refactored code, the K-NN produces the token-level scores by default, but only the sign changes are used to determine the applicable epochs. In this and other cases where token-level labels are not available, or not applicable for the analysis, just use random or constant labels as input for `--test_seq_labels_file`. Note that `--mode "seq_labeling_fine_tune"` is for supervised sequence-labeling, and thus in standard usage is only applicable for the uniCNN+BERT+S* model.
- To update the support set, we re-build the K-NN database structures by re-caching the distances over the additional, or different, exemplar vectors, and then we can use the K-NN inference script for the already trained K-NN weights by simply changing the input path to the database structures.

For the sentiment data, we primarily analyze uniCNN+BERT while changing the data, to further investigate the behavior on domain-shifted and out-of-domain sets. The primary thing to be mindful here is the choice of dataset, the paths of which are determined by the environment variables. Note that the original and revised datasets are separated by directories as in the original works. Given the large number of variations, rather than flatten the scripts by dataset splits, in some cases, the applicable environment variables need to be commented/uncommented depending on the chosen dataset run. To avoid confusion, we would recommend first reading the original publications and reviewing the data directory structure of the original repos [(noted in README_dependencies.md)](README_dependencies.md), and then review our pre-processing scripts (including the creation of the 'disjoint' sets), before reviewing the applicable model and analysis scripts.  


# FCE Grammatical Error Detection Task

## Data

First, we preprocess the FCE data, available at [https://ilexir.co.uk/datasets/index.html](https://ilexir.co.uk/datasets/index.html) with our standard format: [fce_data/data/fce_data_init.sh](fce_data/data/fce_data_init.sh)

Note that this format is as follows, for each document (with a single space between the document-level label and the tokens of the document):

```
0|1 Document text
```

The format of the corresponding labels files are as follows, for each document:

```
0|1 for each token in the document
```

For the experiments with additional newswire text, we also process samples from the news-oriented One Billion Word Benchmark dataset (Chelba et al. 2013): [fce_data/data/out_of_domain_news_data_init.sh](fce_data/data/out_of_domain_news_data_init.sh). This data is available [here](https://drive.google.com/file/d/11MLo5xSyhKLbFoaXFbzOlzMJ7lSK8QKp/view?usp=sharing).

## uniCNN+BERT

Train the base sentence-level model and evaluate at the token-level using the proposed approach: [fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel.sh](fce_data/in_domain/base_sentence_level_model/base_model__train_sentlevel.sh)

Demonstrate how to produce aggregate feature summaries, as proposed in the paper: [fce_data/in_domain/base_sentence_level_model/base_model__grammar_features.sh](fce_data/in_domain/base_sentence_level_model/base_model__grammar_features.sh)

Generate the exemplar vectors and then cache the distances between the query and the support set: [fce_data/in_domain/base_sentence_level_model/base_model__cache_exemplar_vectors_to_disk.sh](fce_data/in_domain/base_sentence_level_model/base_model__cache_exemplar_vectors_to_disk.sh)

Train and evaluate the K-NN models: [fce_data/in_domain/base_sentence_level_model/base_model__linear_exa_train_eval.sh](fce_data/in_domain/base_sentence_level_model/base_model__linear_exa_train_eval.sh)

Next, we consider the domain-shifted/out-of-domain augmented data sets.

Generate the exemplar vectors for the Google News data that augments the FCE training set and the test set. We also then cache the distances between the query and the support set, where the query and support set include the FCE data AND the Google News data: [fce_data/out_of_domain/base_model/base_model__cache_exemplar_vectors_to_disk_ood_google_50kdata_as_db_update.sh](fce_data/out_of_domain/base_model/base_model__cache_exemplar_vectors_to_disk_ood_google_50kdata_as_db_update.sh)  

Separately, we then also cache the distances between the query and the support set, where the query is the FCE+2k News test set and the support set is just the original FCE training set: [fce_data/out_of_domain/base_model/base_model__cache_exemplar_vectors_to_disk_ood_google_data.sh](fce_data/out_of_domain/base_model/base_model__cache_exemplar_vectors_to_disk_ood_google_data.sh)

Next, we evaluate the K-NN first on the FCE+2k News test set using the original FCE training set as the support. For reference, here we also evaluate the zero-shot sequence labeling effectiveness of the original model (uniCNN+BERT) on the FCE+2k News test set. [fce_data/out_of_domain/base_model/base_model__eval_ood_google_data.sh](fce_data/out_of_domain/base_model/base_model__eval_ood_google_data.sh)

Finally, we evaluate the K-NN on the FCE+2k News test set using the *updated support set* consisting of the FCE training set augmented with the 50k News data: [fce_data/out_of_domain/base_model/base_model__eval_ood_google_data_50kdata_as_db_update.sh](fce_data/out_of_domain/base_model/base_model__eval_ood_google_data_50kdata_as_db_update.sh)

## uniCNN+BERT+mm

Next, we fine-tune the base uniCNN+BERT with the min-max constraint. The base model is already rather effective, but these experiments illustrate how we can bias the token-level contributions with priors for a given task. Here the goal is to increase the precision of the token-level predictions, where we encourage sparsity in the class 1 predictions with the assumption that such sentences also have at least 1 token that is of class 0.

First, starting with the uniCNN+BERT model as the initial weights, we fine-tune the CNN parameters using the sentence-level labels: [fce_data/in_domain/mm_model/mm_model__min_max_finetune.sh](fce_data/in_domain/mm_model/mm_model__min_max_finetune.sh)

Next, we generate the exemplar vectors and then cache the distances between the query and the support set: [fce_data/in_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk.sh](fce_data/in_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk.sh)

Train and evaluate the K-NN models: [fce_data/in_domain/mm_model/mm_model__linear_exa_train_eval.sh](fce_data/in_domain/mm_model/mm_model__linear_exa_train_eval.sh)

Next, we consider the domain-shifted/out-of-domain augmented data sets.

Generate the exemplar vectors for the Google News data that augments the FCE training set and the test set. We also then cache the distances between the query and the support set, where the query and support set include the FCE data AND the Google News data: [fce_data/out_of_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk_ood_google_50kdata_as_db_update.sh](fce_data/out_of_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk_ood_google_50kdata_as_db_update.sh)  

Separately, we then also cache the distances between the query and the support set, where the query is the FCE+2k News test set and the support set is just the original FCE training set: [fce_data/out_of_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk_ood_google_data.sh](fce_data/out_of_domain/mm_model/mm_model__cache_exemplar_vectors_to_disk_ood_google_data.sh)

Next, we evaluate the K-NN first on the FCE+2k News test set using the original FCE training set as the support. For reference, here we also evaluate the zero-shot sequence labeling effectiveness of the original model (uniCNN+BERT+mm) on the FCE+2k News test set. [fce_data/out_of_domain/mm_model/mm_model__eval_ood_google_data.sh](fce_data/out_of_domain/mm_model/mm_model__eval_ood_google_data.sh)

Finally, we evaluate the K-NN on the FCE+2k News test set using the *updated support set* consisting of the FCE training set augmented with the 50k News data: [fce_data/out_of_domain/mm_model/mm_model__eval_ood_google_data_50kdata_as_db_update.sh](fce_data/out_of_domain/mm_model/mm_model__eval_ood_google_data_50kdata_as_db_update.sh)

## uniCNN+BERT+S*

Next, we fine-tune the base uniCNN+BERT with token-level labels to create a fully-supervised sequence labeler. The pipeline is similar to the above, but some alternate and additional command-line flags provide access to the token-level signals in training and from the support set.

First, starting with the uniCNN+BERT model as the initial weights, we fine-tune the CNN parameters using the token-level labels: [fce_data/in_domain/supervised_model/supervised_model__supervised_finetune.sh](fce_data/in_domain/supervised_model/supervised_model__supervised_finetune.sh)

Next, we generate the exemplar vectors and then cache the distances between the query and the support set: [fce_data/in_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk.sh](fce_data/in_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk.sh)

Train and evaluate the K-NN models: [fce_data/in_domain/supervised_model/supervised_model__linear_exa_train_eval.sh](fce_data/in_domain/supervised_model/supervised_model__linear_exa_train_eval.sh)

Next, we consider the domain-shifted/out-of-domain augmented data sets.

Generate the exemplar vectors for the Google News data that augments the FCE training set and the test set. We also then cache the distances between the query and the support set, where the query and support set include the FCE data AND the Google News data: [fce_data/out_of_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk_ood_google_data_50kdata_as_db_update.sh](fce_data/out_of_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk_ood_google_data_50kdata_as_db_update.sh)  

Separately, we then also cache the distances between the query and the support set, where the query is the FCE+2k News test set and the support set is just the original FCE training set: [fce_data/out_of_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk_ood_google_data.sh](fce_data/out_of_domain/supervised_model/supervised_model__cache_exemplar_vectors_to_disk_ood_google_data.sh)

Next, we evaluate the K-NN first on the FCE+2k News test set using the original FCE training set as the support. For reference, here we also evaluate the zero-shot sequence labeling effectiveness of the original model (uniCNN+BERT+S*) on the FCE+2k News test set. [fce_data/out_of_domain/supervised_model/supervised_model__eval_ood_google_data.sh](fce_data/out_of_domain/supervised_model/supervised_model__eval_ood_google_data.sh)

Finally, we evaluate the K-NN on the FCE+2k News test set using the *updated support set* consisting of the FCE training set augmented with the 50k News data: [fce_data/out_of_domain/supervised_model/supervised_model__eval_ood_google_data_50kdata_as_db_update.sh](fce_data/out_of_domain/supervised_model/supervised_model__eval_ood_google_data_50kdata_as_db_update.sh)

## Additional Models

For reference, we also train two of the zero-shot models with the 50k News data + FCE data. This provides a point of comparison for examining the effectiveness of training with out-of-domain/domain-shifted data vs. just updating the database with such data via matching. The matching results are above; the `uniCNN+BERT+news50k` and `uniCNN+BERT+mm+news50k` models are those that directly train with the additional data. These two models are identical to uniCNN+BERT and uniCNN+BERT+mm, respectively, only differing in the training data. Train and evaluate these models with the scripts above for uniCNN+BERT and uniCNN+BERT+mm by replacing the training data with `${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_train_size50000.txt`. The corresponding labels file (for the min-max loss and building the support set data structures) is as follows: `${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_train_labels_size50000.txt`.

# Sentiment Detection Task

## Data

First, we preprocess the locally re-edited data for use in the experiments of this section and the next section: [sentiment_data/data/locally_re-edited_sentiment_data_init.sh](sentiment_data/data/locally_re-edited_sentiment_data_init.sh).

We also preprocess the test set of SemEval-2017 Task 4a: [sentiment_data/data/semeval_2017_data_init.sh](sentiment_data/data/semeval_2017_data_init.sh).

For the purposes of replication, since these preprocessing scripts are relatively involved and may require modifications if the original data and/or directory structures are changed in the future, you can download the preprocessed data [here](https://drive.google.com/file/d/11MLo5xSyhKLbFoaXFbzOlzMJ7lSK8QKp/view?usp=sharing). The original datasets retain their original licenses.

## Models

We demonstrate learning and evaluation of the model trained on 3.4k original reviews, the label **ORIG. (3.4k)** in the paper, using the uniCNN+BERT model. The remaining experiments can then be run by changing the training and/or evaluation data, and/or other parameter variations demonstrated in the sections above. Note that aside from the different data, the only two primary differences with the uniCNN+BERT model used in these experiments compared to that used in the FCE experiments are the following:
- Reflecting the longer length of the multi-document reviews, rather than the single sentences of the FCE dataset, we allow a longer max input length of 350. Use the option `--max_length 350`.
- The input is un-tokenized, so we use the option `--input_is_untokenized`.

(to be added shortly)

# Annotation Detection Task/Analysis

(to be added shortly)

---

*For internal use: data hash: zero_2019_2021_v6*

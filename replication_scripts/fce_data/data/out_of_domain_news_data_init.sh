################################################################################
#######  Convert the public news-oriented One Billion Word Benchmark
#######  dataset (Chelba et al. 2013) to our standard 'binaryevalformat' format
################################################################################


################################################################################
#######  First, we need to sample the original dataset. This section is
#######  provided for reference, but the actual files used are available for
#######  public download via the link in README_replication_main_text.md.
################################################################################

# Note that the following input file was derived from the original google1b dataset.
# Samples from it are used in the train/dev/test splits created below. This file is available for download via the link in README_replication_main_text.md.
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/google_1b_combined.binaryevalformat.sample1000000.txt


SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"
OUTPUT_DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments

python -u ${REPO_DIR}/code/data/fce/create_splits_from_binaryevalformat_data.py \
--input_binaryevalformat_file ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/google_1b_combined.binaryevalformat.sample1000000.txt \
--output_train_sizes "50000" \
--output_dev_size 2000 \
--output_test_size 2000 \
--seed_value 1776 \
--output_dir ${OUTPUT_DATA_DIR} \
--output_filename_prefix "google_1b_combined.binaryevalformat" \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL}

#Number of sentences with filtered tokens: 738 out of 1000000


################################################################################
#######  Input and processed files available for download
################################################################################

# The zip archive available via the link in README_replication_main_text.md contains the following:
#
# The following input file (for the script above, but otherwise not used):
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/google_1b_combined.binaryevalformat.sample1000000.txt
#
# and the following output files from the script above and subsequently used in the experiments:
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_train_size50000.txt
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_train_labels_size50000.txt
#
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_dev_size2000.txt
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_dev_labels_size2000.txt
#
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_test_size2000.txt
# ${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/google_1b_combined.binaryevalformat_test_labels_size2000.txt
#
# These then need to be conatenated to the FCE data, as shown below. (The license
# of the FCE data precludes direct distribution.)


################################################################################
#######  Concatenate the news (Y=-1) processed files to the FCE datasets
################################################################################

## concatenate with FCE training:
DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
COMBINED_DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/cat_with_fce
ADD_DATA_SIZE=50000

cat ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt ${OUTPUT_DATA_DIR}/google_1b_combined.binaryevalformat_train_size${ADD_DATA_SIZE}.txt > ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_train_size${ADD_DATA_SIZE}.txt
cat ${DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt ${OUTPUT_DATA_DIR}/google_1b_combined.binaryevalformat_train_labels_size${ADD_DATA_SIZE}.txt > ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_train_labels_size${ADD_DATA_SIZE}.txt

### also, cat FCE dev, test
for SPLIT_NAME in "dev" "test";
do
  OUTPUT_DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments
  DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
  COMBINED_DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/cat_with_fce

  cat ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt ${OUTPUT_DATA_DIR}/google_1b_combined.binaryevalformat_${SPLIT_NAME}_size2000.txt > ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_size2000.txt
  cat ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt ${OUTPUT_DATA_DIR}/google_1b_combined.binaryevalformat_${SPLIT_NAME}_labels_size2000.txt > ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_labels_size2000.txt
done

################################################################################
#######  Convert public FCE error detection data to our standard 'binaryevalformat' format
################################################################################

DATA_DIR=UPDATE_WITH_YOUR_PATH/projects/mltagger/data/fce-error-detection/tsv
OUTPUT_DATA_DIR=UPDATE_WITH_YOUR_PATH/projects/mltagger/data/fce_formatted
REPO_DIR=UPDATE_WITH_YOUR_PATH

for SPLIT_NAME in "train" "dev" "test";
do
python ${REPO_DIR}/code/data/fce/mltagger_format_to_binaryevalformat.py \
--input_mltagger_format_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.tsv \
--output_identification_file ${OUTPUT_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--output_identification_labels_file ${OUTPUT_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt
done


# Number of lines: 28731
# Number of lines: 2222
# Number of lines: 2720

################################################################################
#######  split FCE validation (for semi-supervised tagging)
####### This is the sample split created for the experiments demonstrating
####### setting the decision threshold with a small number of token-labeled
####### sentences. As we show, the strongest zero-shot models are already quite
####### effective without such access to token-level labels. Unless you
####### are specifically aiming to recreate those experiments, this can be
####### skipped to avoid extraneous data files.
################################################################################

DATA_DIR=UPDATE_WITH_YOUR_PATH/projects/mltagger/data/fce_formatted
REPO_DIR=UPDATE_WITH_YOUR_PATH

mkdir ${DATA_DIR}/sampled_splits

python ${REPO_DIR}/code/data/fce/split_validation_with_labels.py \
--valid_binaryeval_file ${DATA_DIR}/fce-public.dev.original.binaryevalformat.txt \
--valid_labels_file ${DATA_DIR}/fce-public.dev.original.binaryevalformat.labels.txt \
--split_0_size 1000 \
--output_binaryeval_split_file_template ${DATA_DIR}/sampled_splits/fce-public.dev.original.binaryevalformat.txt \
--output_labels_split_file_template ${DATA_DIR}/sampled_splits/fce-public.dev.original.binaryevalformat.labels.txt \
--output_index_file_template ${DATA_DIR}/sampled_splits/fce-public.dev.original.binaryevalformat.txt.sample_idx

# Size of split 0: 1000
# Size of split 1: 1222
#
# $ ls ${DATA_DIR}/sampled_splits
# fce-public.dev.original.binaryevalformat.labels.txt_split0_size1000.txt
# fce-public.dev.original.binaryevalformat.labels.txt_split1_size1222.txt
# fce-public.dev.original.binaryevalformat.txt.sample_idx_split0_size1000.txt
# fce-public.dev.original.binaryevalformat.txt.sample_idx_split1_size1222.txt
# fce-public.dev.original.binaryevalformat.txt_split0_size1000.txt
# fce-public.dev.original.binaryevalformat.txt_split1_size1222.txt

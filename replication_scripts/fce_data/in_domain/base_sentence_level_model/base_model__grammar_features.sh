## Here we provide an example of generating aggregated features on the FCE
## grammar training dataset. This uses the uniCNN+BERT model. This is another
## useful tool/approach for quickly analyzing a dataset. In the use cases here,
## we separate the training set by the known, ground-truth document-level labels and then we run
## the feature extraction, separating features associated with correct
## predictions and those associated with incorrect predictions.


################################################################################
#### Here, we separate class 0 and class 1 from the training set
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"
SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted

CLASS_TO_SEPARATE=0
python -u ${REPO_DIR}/code/data/separate_binaryevalformat_classes.py \
--input_binaryevalformat_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt \
--input_binaryevalformat_labels_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt \
--output_binaryevalformat_file ${DATA_DIR}/separated_classes/fce-public.train.original.binaryevalformat.txt.class${CLASS_TO_SEPARATE}.txt \
--output_binaryevalformat_labels_file ${DATA_DIR}/separated_classes/fce-public.train.original.binaryevalformat.labels.txt.class${CLASS_TO_SEPARATE}.txt \
--class_to_separate ${CLASS_TO_SEPARATE}

CLASS_TO_SEPARATE=1
python -u ${REPO_DIR}/code/data/separate_binaryevalformat_classes.py \
--input_binaryevalformat_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt \
--input_binaryevalformat_labels_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt \
--output_binaryevalformat_file ${DATA_DIR}/separated_classes/fce-public.train.original.binaryevalformat.txt.class${CLASS_TO_SEPARATE}.txt \
--output_binaryevalformat_labels_file ${DATA_DIR}/separated_classes/fce-public.train.original.binaryevalformat.labels.txt.class${CLASS_TO_SEPARATE}.txt \
--class_to_separate ${CLASS_TO_SEPARATE}


################################################################################
#### GRAMMAR -- only pos class from training
#### grammar features for the training sentences with errors
#### CLASS_TO_SEPARATE=1
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted/separated_classes
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/features_v2"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="15"

SPLIT_NAME="train"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CLASS_TO_SEPARATE=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "features" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt.class${CLASS_TO_SEPARATE}.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt.class${CLASS_TO_SEPARATE}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 30 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/not_used \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/not_used \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/not_used \
--fce_eval \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability 0.5 \
--output_neg_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.neg_features.sorted.v1.txt \
--output_pos_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.pos_features.sorted.v1.txt \
--output_neg_sentence_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.neg_sentence_features.sorted.v1.txt \
--output_pos_sentence_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.pos_sentence_features.sorted.v1.txt


# This is run over all sentences of *ground-truth* class 1 (pos). In the case
# of the FCE set, these are sentences annotated as having grammar/style errors.
# Among these,
# the 'neg' files contain ngrams/sentences the model predicts as class 0 (based
# on the sentence-level prediction from the fully-connected layer). These
# are all wrong predictions, at least according to the ground-truth labels.
# Similarly, the 'neg' files contain ngrams/sentences the model predicts as class 1.
# These are all correct predictions, at least according to the ground-truth labels.
# Within the ngram files (--output_neg_features_file and --output_pos_features_file),
# we sort by total logit (search for 'sorted by total logit')
# and normalized by the frequency (search for 'normalized by occurrence').
# For example, the result in the appendix of the paper looks
# as follows in the raw output:
# in "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.pos_features.sorted.v1.txt
####ngram size: 1; normalized by occurrence
# wating: 22.499080657958984 || Occurrences: 1
# noize: 21.942635536193848 || Occurrences: 1
# exitation: 21.49026346206665 || Occurrences: 1
# exitement: 21.174572467803955 || Occurrences: 1
# toe: 20.13607406616211 || Occurrences: 1
# fite: 19.99588680267334 || Occurrences: 1
# ofer: 19.970499992370605 || Occurrences: 2
# n: 19.67680950164795 || Occurrences: 5
# intents: 18.560322761535645 || Occurrences: 1
# wit: 17.733697414398193 || Occurrences: 2
# defences: 17.51922035217285 || Occurrences: 1
# meannes: 17.475926637649536 || Occurrences: 1
# baying: 17.34419012069702 || Occurrences: 1
# saing: 17.113635063171387 || Occurrences: 2
# dipends: 16.99739408493042 || Occurrences: 1
# lair: 16.713510513305664 || Occurrences: 2
# ...

# Similarly, for sentences (--output_neg_sentence_features_file and --output_pos_sentence_features_file)
# we sort by the total score (search for '####Unnormalized scores'),
# and when normalizing by the sentence length (search for '####Mean (length normalized) scores').




################################################################################
#### GRAMMAR -- only neg class from training
#### grammar features for the training sentences without errors
#### CLASS_TO_SEPARATE=0
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted/separated_classes
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/features_v2"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="15"

SPLIT_NAME="train"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CLASS_TO_SEPARATE=0
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "features" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt.class${CLASS_TO_SEPARATE}.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt.class${CLASS_TO_SEPARATE}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 30 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/not_used \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/not_used \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/not_used \
--fce_eval \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability 0.5 \
--output_neg_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.neg_features.sorted.v1.txt \
--output_pos_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.pos_features.sorted.v1.txt \
--output_neg_sentence_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.neg_sentence_features.sorted.v1.txt \
--output_pos_sentence_features_file "${OUTPUT_DIR}"/fce-public.train.original.binaryevalformat.class${CLASS_TO_SEPARATE}.pos_sentence_features.sorted.v1.txt

# The convention is the same as described in the comment for class 1 above, except now the input
# only includes class 0, which for the FCE set, are sentences that are deemed
# correct/adequate/no errors by the annotators.

################################################################################
#### In these experiments, we aim to predict the feature of interest---in this
#### case the tokens that are revised by the annotators to flip the label of
#### the review. We will occasionally refer to this as 'domain' prediction below.
#### Note that we are NOT aiming to predict sentiment, as in the
#### main task. (To pre-empt possible confusion, note that we do subset some of the 'domain'
#### prediction sets by sentiment for analysis purposes below, but the task is still to predict the 'domain'.)
################################################################################


################################################################################
#### First, train the uniCNN+BERT model to predict original vs. revised
#### reviews (NOT sentiment). The training set is derived from the paired
#### (source-target) counterfactually-augemented training set, consisting of
#### 3414 reviews.
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0


VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}"

OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

TRAIN_DATA_DIR=${DATA_DIR}/combined/paired/domain
DEV_DATA_DIR=${TRAIN_DATA_DIR}
TRAIN_DATA_INPUT_NAME="train.paired.domain_binaryevalformat.txt" # wc -l: 3414
DEV_DATA_INPUT_NAME="dev.paired.domain_binaryevalformat.txt" # wc -l: 490


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${SERVER_DRIVE_PATH_PREFIX}/data/general/GoogleNews-vectors-negative300.bin \
--training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
--dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--save_model \
--epoch 60 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${MODEL_DIR}" \
--score_vals_file "${OUTPUT_DIR}"/train.dev_score_vals.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB}  \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/train.log.txt

# epoch: 36 / dev_acc: 0.7938775510204081
# max dev acc: 0.7938775510204081


################################################################################
#######  TEST (review level) -- on domain prediction data
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}

## Comment/uncomment applicable variables to get the desired data split. Additional notes appear below after the main code block.


# #SPLIT_NAME="dev"
# SPLIT_NAME="test"
# DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"
#
# #DATA_DESCR="only_orig" # class 0
# DATA_DESCR="only_new" # class 1
# #DATA_DESCR="paired"
# EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain
# EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.txt"
# SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt"

## the following is for the data that is further subdivided by sentiment
SPLIT_NAME="test"
DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

#DATA_DESCR="only_negative_sentiment"
#DATA_DESCR="only_positive_sentiment"

#DATA_DESCR="only_negative_sentiment_only_new"
#DATA_DESCR="only_negative_sentiment_only_orig"

#DATA_DESCR="only_positive_sentiment_only_new"
DATA_DESCR="only_positive_sentiment_only_orig"
EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain/separated_by_sentiment
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.paired.domain_binaryevalformat.${DATA_DESCR}.txt"
SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.paired.domain_binaryevalformat.sequence_labels.${DATA_DESCR}.txt"


EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_domain.${DATA_DESCR}.split_${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_domain.${DATA_DESCR}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt

### We include the results in-line below to aid in making the correspondence between the environment variables and the dataset splits and the results in the paper (see Table 13 in arXiv v6).

DATA_DESCR="paired"
SPLIT_NAME="dev"
  # (Accuracy from random prediction (only for debugging purposes): 0.47346938775510206)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.5)
  # Ground-truth Stats: Number of instances with class 1: 245 out of 490
# test acc: 0.7938775510204081  # same as produced above from training (note this is the 'dev' split)

DATA_DESCR="paired"
SPLIT_NAME="test"

  # (Accuracy from random prediction (only for debugging purposes): 0.4969262295081967)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.5)
  # Ground-truth Stats: Number of instances with class 1: 488 out of 976
# test acc: 0.7961065573770492

DATA_DESCR="only_new"
SPLIT_NAME="test"
  # (Accuracy from random prediction (only for debugging purposes): 0.5266393442622951)
  # (Accuracy from all 1's prediction (only for debugging purposes): 1.0)
  # Ground-truth Stats: Number of instances with class 1: 488 out of 488
# test acc: 0.805327868852459

DATA_DESCR="only_orig"
SPLIT_NAME="test"
  # (Accuracy from random prediction (only for debugging purposes): 0.5)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.0)
  # Ground-truth Stats: Number of instances with class 1: 0 out of 488
# test acc: 0.7868852459016393


DATA_DESCR="only_negative_sentiment"
  # (Accuracy from random prediction (only for debugging purposes): 0.5081967213114754)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.5020491803278688)
  # Ground-truth Stats: Number of instances with class 1: 245 out of 488
# test acc: 0.8401639344262295

DATA_DESCR="only_positive_sentiment"
  # (Accuracy from random prediction (only for debugging purposes): 0.49385245901639346)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.4979508196721312)
  # Ground-truth Stats: Number of instances with class 1: 243 out of 488
# test acc: 0.7520491803278688

DATA_DESCR="only_negative_sentiment_only_new"
  # (Accuracy from random prediction (only for debugging purposes): 0.49387755102040815)
  # (Accuracy from all 1's prediction (only for debugging purposes): 1.0)
  # Ground-truth Stats: Number of instances with class 1: 245 out of 245
# test acc: 0.8408163265306122

DATA_DESCR="only_negative_sentiment_only_orig"
  # (Accuracy from random prediction (only for debugging purposes): 0.5390946502057613)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.0)
  # Ground-truth Stats: Number of instances with class 1: 0 out of 243
# test acc: 0.8395061728395061

DATA_DESCR="only_positive_sentiment_only_new"
  # (Accuracy from random prediction (only for debugging purposes): 0.5185185185185185)
  # (Accuracy from all 1's prediction (only for debugging purposes): 1.0)
  # Ground-truth Stats: Number of instances with class 1: 243 out of 243
# test acc: 0.7695473251028807

DATA_DESCR="only_positive_sentiment_only_orig"
  # (Accuracy from random prediction (only for debugging purposes): 0.5510204081632653)
  # (Accuracy from all 1's prediction (only for debugging purposes): 0.0)
  # Ground-truth Stats: Number of instances with class 1: 0 out of 245
# test acc: 0.7346938775510204

# When making comparisons, keep in mind that these subsets have different numbers of sentences.


################################################################################
#######  ZERO -- with sequence labels
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}

SPLIT_NAME="dev"
#SPLIT_NAME="test"
DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

#DATA_DESCR="only_orig" # class 0
#DATA_DESCR="only_new" # class 1
DATA_DESCR="paired"
EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.txt"
SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt"

EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DATA_OUTPUT_SUBSET="onlycorrect"
#DATA_OUTPUT_SUBSET="all_sentences"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero_sentiment" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DATA_DESCR}.split_${SPLIT_NAME}.maxdevepoch.zerorun.${DATA_OUTPUT_SUBSET}.txt \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot.viz.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.detection.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.txt \
--detection_offset 0 \
--fce_eval \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/${SEQ_LABELS_INPUT_NAME} \
--only_visualize_correct_predictions >"${OUTPUT_LOG_DIR}"/zero_shot.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.log.txt

# The output HTML file --visualization_out_file visualizes the token-level predictions. Tokens highlighted in red corresponds to those with a positive detection (y_n=1), and all other tokens (y_n=-1) are in blue. You can limit the file to only include documents with incorrect document-level predictions by including the flag
# --only_visualize_missed_predictions
# and similarly, you can limit the file to only include documents with correct document-level predictions by including the flag
# --only_visualize_correct_predictions
# If not including those flags, all documents are included in the HTML file.

## NOTE: In our research code, 'Sentiment' is hard-coded into the HTML output, but the 'Softmax global predictions' correspond to the classes of the given dataset (here, annotator domain).


################################################################################
#### cat dev and test
#### also, cat train, dev, and test
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"


EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain
OUTPUT_CAT_DIR=${DATA_DIR}/combined/paired/domain/concatenated/
mkdir ${OUTPUT_CAT_DIR}

#DATA_DESCR="only_orig" # class 0
DATA_DESCR="only_new" # class 1

# cat train, dev, test
# text:
cat ${EVAL_DATA_DIR}/train.${DATA_DESCR}.domain_binaryevalformat.txt ${EVAL_DATA_DIR}/dev.${DATA_DESCR}.domain_binaryevalformat.txt ${EVAL_DATA_DIR}/test.${DATA_DESCR}.domain_binaryevalformat.txt > ${OUTPUT_CAT_DIR}/cat_train_dev_test.${DATA_DESCR}.domain_binaryevalformat.txt
# seq labels:
cat ${EVAL_DATA_DIR}/train.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt ${EVAL_DATA_DIR}/dev.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt ${EVAL_DATA_DIR}/test.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt > ${OUTPUT_CAT_DIR}/cat_train_dev_test.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt

# cat just dev and test
# text:
cat ${EVAL_DATA_DIR}/dev.${DATA_DESCR}.domain_binaryevalformat.txt ${EVAL_DATA_DIR}/test.${DATA_DESCR}.domain_binaryevalformat.txt > ${OUTPUT_CAT_DIR}/cat_dev_test.${DATA_DESCR}.domain_binaryevalformat.txt
# seq labels:
cat ${EVAL_DATA_DIR}/dev.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt ${EVAL_DATA_DIR}/test.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt > ${OUTPUT_CAT_DIR}/cat_dev_test.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt


################################################################################
#### Generate features for each of the classes -- concatenated data
#### We did not use this in the paper, but this is just an example of how we
#### could check for unexpected distributional differences across dataset
#### splits (seen in training vs. eval and/or combinations thereof).
#### For the results on the dev set used in the paper, see the next section.
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/features_v2"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

#DATA_DESCR="only_orig" # class 0 ("neg")
DATA_DESCR="only_new" # class 1 ("pos")
EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain/concatenated

SPLIT_NAME="cat_train_dev_test"  # this is ALL of the data (including that seen during training)
#SPLIT_NAME="cat_dev_test"  # just dev and test
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt"

EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "features" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/${EVAL_DATA_INPUT_NAME} \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/not_used.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--fce_eval \
--dropout_probability ${DROPOUT_PROB} \
--output_neg_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.neg_features.sorted.v1.txt \
--output_pos_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.pos_features.sorted.v1.txt \
--output_neg_sentence_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.neg_sentence_features.sorted.v1.txt \
--output_pos_sentence_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.pos_sentence_features.sorted.v1.txt


################################################################################
#### Generate features for each of the classes -- dev set
#### arXiv v6: Table 14 and Table 15
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/features_v2/only_dev"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

#DATA_DESCR="only_orig" # class 0 ("neg")
DATA_DESCR="only_new" # class 1 ("pos")
#EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain/concatenated
EVAL_DATA_DIR=${DATA_DIR}/combined/paired/domain

SPLIT_NAME="dev"
#SPLIT_NAME="cat_train_dev_test"  # this is ALL of the data (including that seen during training)
#SPLIT_NAME="cat_dev_test"  # just dev and test
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.${DATA_DESCR}.domain_binaryevalformat.sequence_labels.txt"

EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "features" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/${EVAL_DATA_INPUT_NAME} \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/not_used.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--fce_eval \
--dropout_probability ${DROPOUT_PROB} \
--output_neg_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.neg_features.sorted.v1.txt \
--output_pos_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.pos_features.sorted.v1.txt \
--output_neg_sentence_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.neg_sentence_features.sorted.v1.txt \
--output_pos_sentence_features_file "${OUTPUT_DIR}"/${EVAL_DATA_INPUT_NAME}.pos_sentence_features.sorted.v1.txt




################################################################################
#######  TEST (review level) -- on domain prediction data -- CONTRAST SETS
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/contrast_sets"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

#SPLIT_NAME="dev"
SPLIT_NAME="test"

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/constrast_sets/contrast-sets-master/IMDb/data/binaryevalformat/domain"

DATA_DESCR="all"
DATA_DESCR="only_orig"
DATA_DESCR="only_new"
DATA_DESCR="only_neg"
DATA_DESCR="only_pos"

DATA_DESCR="only_orig_only_neg"
DATA_DESCR="only_orig_only_pos"
DATA_DESCR="only_new_only_neg"
DATA_DESCR="only_new_only_pos"

EVAL_DATA_DIR=${DATA_DIR}
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.binaryevalformat.domain.${DATA_DESCR}.txt"
SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.binaryevalformat.domain_diffs_sequence_labels.${DATA_DESCR}.txt"


EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_domain.${DATA_DESCR}.split_${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_domain.${DATA_DESCR}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt

echo ${SPLIT_NAME}
echo ${DATA_DESCR}
echo "${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_domain.${DATA_DESCR}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt

### We include the results in-line below to aid in making the correspondence between the environment variables and the dataset splits and the results in the paper (see Table 17 in arXiv v6).

###
# SPLIT_NAME="dev"
# DATA_DESCR="all"
# # test acc: 0.74 (note this is the 'dev' split and does not appear in the table)

#######################################
# SPLIT_NAME="test"
#######################################
DATA_DESCR="all"
# test acc: 0.7776639344262295


DATA_DESCR="only_orig"
# test acc: 0.7868852459016393

DATA_DESCR="only_new"
# test acc: 0.7684426229508197

DATA_DESCR="only_neg"
# test acc: 0.7868852459016393

DATA_DESCR="only_pos"
# test acc: 0.7684426229508197


DATA_DESCR="only_orig_only_neg"
# test acc: 0.8395061728395061

DATA_DESCR="only_orig_only_pos"
# test acc: 0.7346938775510204

DATA_DESCR="only_new_only_neg"
# test acc: 0.7346938775510204

DATA_DESCR="only_new_only_pos"
# test acc: 0.8024691358024691

################################################################################
#######  ZERO -- with sequence labels -- CONTRAST SETS
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_predict_domain_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_fw0_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/contrast_sets/zero"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

SPLIT_NAME="dev"
#SPLIT_NAME="test"

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/constrast_sets/contrast-sets-master/IMDb/data/binaryevalformat/domain"

DATA_DESCR="all"
# DATA_DESCR="only_orig"
# DATA_DESCR="only_new"
# DATA_DESCR="only_neg"
# DATA_DESCR="only_pos"
#
# DATA_DESCR="only_orig_only_neg"
# DATA_DESCR="only_orig_only_pos"
# DATA_DESCR="only_new_only_neg"
# DATA_DESCR="only_new_only_pos"

EVAL_DATA_DIR=${DATA_DIR}
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.binaryevalformat.domain.${DATA_DESCR}.txt"
SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.binaryevalformat.domain_diffs_sequence_labels.${DATA_DESCR}.txt"

EPOCH="36"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DATA_OUTPUT_SUBSET="onlycorrect"
#DATA_OUTPUT_SUBSET="all_sentences"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero_sentiment" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DATA_DESCR}.split_${SPLIT_NAME}.maxdevepoch.zerorun.${DATA_OUTPUT_SUBSET}.txt \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot.viz.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.detection.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.txt \
--detection_offset 0 \
--fce_eval \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/${SEQ_LABELS_INPUT_NAME} \
--only_visualize_correct_predictions >"${OUTPUT_LOG_DIR}"/zero_shot.dataprefix_${DATA_DESCR}.${SPLIT_NAME}.epoch${EPOCH}.${DATA_OUTPUT_SUBSET}.log.txt

################################################################################
#### The following uniCNN+BERT is trained on the 3.4k orig data.
################################################################################

################################################################################
#### The following uniCNN+BERT is trained on the 3.4k orig data.
#### First, train at the review-level.
################################################################################


SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0


VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

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

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/orig"

TRAIN_DATA_DIR=${DATA_DIR}/binaryevalformat/train3.4k
DEV_DATA_DIR=${DATA_DIR}/binaryevalformat
TRAIN_DATA_INPUT_NAME="train.orig_3.4k.binaryevalformat.txt"
DEV_DATA_INPUT_NAME="dev.tsv.binaryevalformat.txt"


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

# For this dataset, we simply choose the epoch with the highest document-level accuracy on the dev set, since the classes are balanced
# epoch: 31 / dev_acc: 0.9428571428571428

################################################################################
#######  The following uniCNN+BERT is trained on the 3.4k orig data.
#######  TEST (review level) -- on orig and new data
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

#SPLIT_NAME="dev"
SPLIT_NAME="test"
DATA_INPUT_NAME="${SPLIT_NAME}.tsv.binaryevalformat.txt"
#DIR_PREFIX="orig"
DIR_PREFIX="new"
DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat"

EPOCH="31"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt


# this is the test set of the orig data
SPLIT_NAME="test"
DIR_PREFIX="orig"

#test acc: 0.9282786885245902

# this is the test set of the new data (i.e., counterfactually-augmented data)
SPLIT_NAME="test"
DIR_PREFIX="new"

#test acc: 0.8872950819672131


################################################################################
#######  ZERO-shot sequence labeling evaluated against the sequence labels (constructed
#######  from the sentiment diffs) -- orig vs. new
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


#SPLIT_NAME="dev"
SPLIT_NAME="test"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.sequence_labels.txt"
#DIR_PREFIX="orig"
DIR_PREFIX="new"
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"



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
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.maxdevepoch.zerorun.txt \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot.viz.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.detection.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.txt \
--detection_offset 0 \
--fce_eval \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/"${EVAL_SEQ_LABELS_INPUT_NAME}" >"${OUTPUT_LOG_DIR}"/zero_shot.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.log.txt

SPLIT_NAME="dev"
DIR_PREFIX="orig"

# DETECTION OFFSET: 2.200000000000001
# GENERATED:
#         Precision: 0.4897959183673469
#         Recall: 0.10792580101180438
#         F1: 0.17687701520036847
#         F0.5: 0.2868240215118016
#         MCC: 0.21510923802318932
#
#
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.08356330720830114
#         Recall: 0.4255199550309162
#         F1: 0.1396936704188965
#         F0.5: 0.09956596080494542
#         MCC: 0.10391629651524148

SPLIT_NAME="test"
DIR_PREFIX="orig"

# This offset is chosen on the dev set above. Note that this corresponds to
# Table 9 in arXiv v6.
# DETECTION OFFSET: 2.200000000000001
# GENERATED:
#         Precision: 0.45738295318127253
#         Recall: 0.09665144596651445
#         F1: 0.15958115183246072
#         F0.5: 0.26189166895793237
#         MCC: 0.19328031631883913
#
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.08853702051739518
#         Recall: 0.4028411973617453
#         F1: 0.14516866258341712
#         F0.5: 0.10490711624342679
#         MCC: 0.0967074396647551

SPLIT_NAME="test"
DIR_PREFIX="new"


# DETECTION OFFSET: 2.200000000000001
# GENERATED:
#         Precision: 0.4728476821192053
#         Recall: 0.07264957264957266
#         F1: 0.1259481390015876
#         F0.5: 0.22498109402571215
#         MCC: 0.16702807442305942





################################################################################
#### EXEMPLAR EVAL -- Table 10 (arXiv v6), row for Orig. (3.4k)
#### In this version of the code, we need to first cache the distances, train
#### the K-NN, create the archive file of the output, and then run the exemplar auditing decision rules script.
#### See orig3k_model__cache_exemplar_vectors_to_disk.sh and
#### orig3k_model__linear_exa_train_eval.sh for an example. Here, we provide some additional
#### notes on the dataset splits.
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

### Comment/Uncomment the following blocks depending on the desired datasets for evaluation and as the support set
# ################# database
# DB_SPLIT_NAME="train"
# TRAINING_DIR_PREFIX="new"
# #TRAINING_DIR_PREFIX="orig"
# DB_TXT_FILE="train.aligned_binaryevalformat.txt"
# DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${TRAINING_DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"

################# database Orig.+Rev.
DB_SPLIT_NAME="train"
#DB_SPLIT_NAME="dev"
TRAINING_DIR_PREFIX="orig_and_new"
DB_TXT_FILE="${DB_SPLIT_NAME}_paired.tsv.binaryevalformat.txt"
DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/binaryevalformat"

################# query

SPLIT_NAME="test"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.sequence_labels.txt"
#DIR_PREFIX="orig"
DIR_PREFIX="new"
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"


## for each split above, run:
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "generate_exemplar_data" \
...

## followed by:
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
...

## followed by (just once and can train for a single epoch just to create a place-holder K-NN if you're only
## interested in reproducing the results for the exemplar auditing decision rules)
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train_linear_exa" \
...

## followed by (using the update flag, --update_knn_model_database, as applicable)
## This will create an archive file (--output_prediction_stats_file) with the
## data structures used in evaluating the decision rules below.
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
...

## followed by
python exa_analysis_rules.py \
--output_prediction_stats_file "created_by_running_eval_linear_exa_above" \
--analysis_type "ExAG"


##### Correspondence to Column Labels of Table 10 (arXiv v6)
##### S=Orig. (1.7k); Test: Orig.
TRAINING_DIR_PREFIX="orig"
DIR_PREFIX="orig"

##### S=Orig. (1.7k); Test: Rev.
TRAINING_DIR_PREFIX="orig"
DIR_PREFIX="new"

##### S=Rev. (1.7k); Test: Orig.
TRAINING_DIR_PREFIX="new"
DIR_PREFIX="orig"

##### S=Rev. (1.7k); Test: Rev.
TRAINING_DIR_PREFIX="new"
DIR_PREFIX="new"

##### S=Orig. (1.7k)+Rev. (1.7k); Test: Orig.
DB_SPLIT_NAME="train"
TRAINING_DIR_PREFIX="orig_and_new"
DIR_PREFIX="orig"

##### S=Orig. (1.7k)+Rev. (1.7k); Test: Rev.
DB_SPLIT_NAME="train"
TRAINING_DIR_PREFIX="orig_and_new"
DIR_PREFIX="new"


################################################################################
#######  TEST (review-level) -- on SemEval2017 balance
#######  Here, we get the review-level accuracy on the out-of-domain Twitter data
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


SPLIT_NAME="test"
DIR_PREFIX="balanced"
DATA_INPUT_NAME="SemEval2017-task4-${SPLIT_NAME}.subtask-A.english.txt.binaryevalformat.${DIR_PREFIX}.txt"


DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/SemEval2017-task4-test/binaryevalformat"

EPOCH="31"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt


# test acc: 0.7781052631578947


################################################################################
#######  TEST (review level) -- on Contrast Set test
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/contrast_sets"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


SPLIT_NAME="test"
#DIR_PREFIX="original"
DIR_PREFIX="contrast"
DATA_INPUT_NAME="${SPLIT_NAME}_${DIR_PREFIX}.tsv.binaryevalformat.txt"
DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/constrast_sets/contrast-sets-master/IMDb/data/binaryevalformat"

EPOCH="31"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized >"${OUTPUT_LOG_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.epoch${EPOCH}.log.txt


SPLIT_NAME="test"
DIR_PREFIX="original"

#test acc: 0.9282786885245902

SPLIT_NAME="test"
DIR_PREFIX="contrast"

#test acc: 0.8237704918032787


################################################################################
#######  ZERO-shot sequence labeling evaluated against the sequence labels (constructed
#######  from the sentiment diffs)-- on Contrast Set test
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH


REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.50
MAX_SENT_LEN=350
EXPERIMENT_LABEL=counter_orig3k_unicnnbert_v${VOCAB_SIZE}k_bertlargecased_top4layers_fn${FILTER_NUMS}_dropout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}"

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/contrast_sets/zero"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


SPLIT_NAME="test"
DIR_PREFIX="contrast"
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/constrast_sets/contrast-sets-master/IMDb/data/binaryevalformat"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}_${DIR_PREFIX}.tsv.binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="sentiment_sequence_labels/${SPLIT_NAME}.binaryevalformat.sentiment_diffs_sequence_labels.new.txt"


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


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
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.maxdevepoch.zerorun.txt \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot.viz.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.detection.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.txt \
--detection_offset 0 \
--fce_eval \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/"${EVAL_SEQ_LABELS_INPUT_NAME}" >"${OUTPUT_LOG_DIR}"/zero_shot.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.log.txt

SPLIT_NAME="test"
DIR_PREFIX="contrast"

# See Table 11 of arXiv v6
# DETECTION OFFSET: 2.200000000000001
# GENERATED:
#         Precision: 0.41944847605224966
#         Recall: 0.05096102980074061
#         F1: 0.09088050314465408
#         F0.5: 0.17147264744274357
#         MCC: 0.1265524396363098

# See Table 12 of arXiv v6
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.10600469641060047
#         Recall: 0.33433256921177923
#         F1: 0.1609712611962474
#         F0.5: 0.12277407239525998
#         MCC: 0.07317307756654033

################################################################################
#### Exemplar Auditing decision rules -- CONTRAST SETS
################################################################################

# We can then evaluate the exemplar auditing decision rules over the contrast
# set test set in the same manner as was done with the counterfactually-augmented
# data. Note that the model is the same as
# above, but the the eval set is now the Contrast Sets test set from the previous
# section. Additionally, in the rightmost column of Table 12 (arXiv v6), we also
# include the results with the support set consisting of the paired dev set,
# S=ORIG._DEV+REV._DEV (245+245).

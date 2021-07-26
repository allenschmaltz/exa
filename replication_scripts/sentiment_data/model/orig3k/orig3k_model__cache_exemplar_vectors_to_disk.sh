################################################################################
#### uniCNN+BERT; cache exemplar vectors and distances to disk
#### Note that this model is only trained with the original 3k data.
################################################################################


################################################################################
#### generate exemplar data
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

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

# ################# database
# # this is for the combined orig+new runs
# #SPLIT_NAME="train"
# SPLIT_NAME="dev"
# DIR_PREFIX="orig_and_new"
# EVAL_DATA_INPUT_NAME="${SPLIT_NAME}_paired.tsv.binaryevalformat.txt"
# EVAL_SEQ_LABELS_INPUT_NAME="sequence_labels/${SPLIT_NAME}_paired.tsv.binaryevalformat.sequence_labels.txt"
# EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/binaryevalformat"


# SPLIT_NAME="train"
################# query
SPLIT_NAME="dev"  # this model should only see orig dev for tuning
#SPLIT_NAME="test"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.sequence_labels.txt"
DIR_PREFIX="orig"
#DIR_PREFIX="new" # when eval in other directory, rewrite above (which is used for model dir)
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "generate_exemplar_data" \
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
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--output_exemplar_data_file "${OUTPUT_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt



################################################################################
#### uniCNN+BERT;  -- save exemplar distance structures
#### This is the dev set from the original data, consisting of only 245
#### sentences, which is then split
#### itself into 2 sets for training and dev of the K-NN. The 'database' here
#### is the original with 1.7k. Note that this is half the data the model was
#### originally trained on. The reason for this was to keep constant the amount
#### of data in the database w.r.t. orig and new, as in the tables with the
#### inference-time decision rules, which we just keep the same here for the
#### K-NN, as well. (The K-NN training set, K-NN dev set, and the support set are all relatively
#### small in this case, but turn out to be sufficient for learning the 3 parameters of
#### the K-NN up to the effectiveness of the original model against the
#### ground-truth.)
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

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
# for output on second drive -- this is for the exemplar text vector for orig dev; otherwise, the files are in EXEMPLAR_DIR: these directories are the same if SERVER_DRIVE_PATH_PREFIX==SERVER_SCRATCH_DRIVE_PATH_PREFIX
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

################# database
DB_SPLIT_NAME="train"
#TRAINING_DIR_PREFIX="new"
TRAINING_DIR_PREFIX="orig"
DB_TXT_FILE="train.aligned_binaryevalformat.txt"
DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${TRAINING_DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"

# ################# database Orig.+Rev.
# DB_SPLIT_NAME="train"
# #DB_SPLIT_NAME="dev"
# TRAINING_DIR_PREFIX="orig_and_new"
# DB_TXT_FILE="${DB_SPLIT_NAME}_paired.tsv.binaryevalformat.txt"
# DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/binaryevalformat"

################# query
SPLIT_NAME="dev"
#SPLIT_NAME="test"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.sequence_labels.txt"
DIR_PREFIX="orig"
#DIR_PREFIX="new" # when eval in other directory, rewrite above (which is used for model dir)
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/db_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}_${DIR_PREFIX}_${SPLIT_NAME}_dist_dir"
mkdir -p "${DISTANCE_DEV_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
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
--bert_gpu -1 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--exemplar_sentences_database_file ${DB_DATA_DIR}/${DB_TXT_FILE} \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/exemplar.dataprefix_${TRAINING_DIR_PREFIX}.${DB_SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_data_query_file "${SECOND_EXEMPLAR_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 10 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_database_data_structure \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--create_train_eval_split_from_query_for_knn \
--binomial_sample_p 0.5 \
--query_train_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-train-split_chunks_ids.pt" \
--query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt




################################################################################
#### uniCNN+BERT;  -- save exemplar distance structures
#### Similarly, here we save the test sets, but note here we do not need to
#### save a query-train/query-eval split.
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

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
# for output on second drive -- this is for the exemplar text vector for orig dev; otherwise, the files are in EXEMPLAR_DIR: these directories are the same if SERVER_DRIVE_PATH_PREFIX==SERVER_SCRATCH_DRIVE_PATH_PREFIX
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

################# database
DB_SPLIT_NAME="train"
#TRAINING_DIR_PREFIX="new"
TRAINING_DIR_PREFIX="orig"
DB_TXT_FILE="train.aligned_binaryevalformat.txt"
DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${TRAINING_DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"

# ################# database Orig.+Rev.
# DB_SPLIT_NAME="train"
# #DB_SPLIT_NAME="dev"
# TRAINING_DIR_PREFIX="orig_and_new"
# DB_TXT_FILE="${DB_SPLIT_NAME}_paired.tsv.binaryevalformat.txt"
# DB_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/binaryevalformat"

################# query
#SPLIT_NAME="dev"
SPLIT_NAME="test"
EVAL_DATA_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.txt"
EVAL_SEQ_LABELS_INPUT_NAME="${SPLIT_NAME}.aligned_binaryevalformat.sequence_labels.txt"
DIR_PREFIX="orig"
#DIR_PREFIX="new" # when eval in other directory, rewrite above (which is used for model dir)
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/db_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}_${DIR_PREFIX}_${SPLIT_NAME}_dist_dir"
mkdir -p "${DISTANCE_DEV_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
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
--bert_gpu -1 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--exemplar_sentences_database_file ${DB_DATA_DIR}/${DB_TXT_FILE} \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/exemplar.dataprefix_${TRAINING_DIR_PREFIX}.${DB_SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 10 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_database_data_structure \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt


# SPLIT_NAME="test"
# DIR_PREFIX="orig"
#
#
# SPLIT_NAME="test"
# DIR_PREFIX="new"

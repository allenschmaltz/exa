################################################################################
#### uniCNN+BERT; cache exemplar vectors and distances to disk
#### Note that this model is only trained with the original 3k data.
#### This is for the out-of-domain SemEval2017 data.
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

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017/exemplar"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p ${OUTPUT_DIR}
mkdir ${OUTPUT_LOG_DIR}


EPOCH="31"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

SPLIT_NAME="test"
DIR_PREFIX="balanced"
DATA_INPUT_NAME="SemEval2017-task4-${SPLIT_NAME}.subtask-A.english.txt.binaryevalformat.${DIR_PREFIX}.txt"

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/SemEval2017-task4-test/binaryevalformat"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "generate_exemplar_data" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
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
--test_seq_labels_file ${DATA_DIR}/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.monolabels.balanced.txt \
--output_exemplar_data_file "${OUTPUT_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt

# 768M    UPDATE_WITH_YOUR_PATH/output/binary/sentiment/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/eval/semeval2017/exemplar/exemplar.dataprefix_balanced.test.epoch31.exemplar_data.txt



################################################################################
#### uniCNN+BERT;  -- save exemplar distance structures
#### Here we save the SemEval2017 test set, for which we do not need to
#### save a query-train/query-eval split. The database is the 1.7k orig training data.
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
# for output on second drive -- this is for the exemplar text vector for semeval2017 test; otherwise, the files are in EXEMPLAR_DIR:
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017/exemplar/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p ${OUTPUT_DIR}
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


################# query
# semeval2017 data:
SPLIT_NAME="test"
DIR_PREFIX="balanced"
DATA_INPUT_NAME="SemEval2017-task4-${SPLIT_NAME}.subtask-A.english.txt.binaryevalformat.${DIR_PREFIX}.txt"

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/SemEval2017-task4-test/binaryevalformat"


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017/exemplar/db_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}_${DIR_PREFIX}_${SPLIT_NAME}_dist_dir"
mkdir -p "${DISTANCE_DEV_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
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
--test_seq_labels_file ${DATA_DIR}/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.monolabels.balanced.txt \
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
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt


# The top 25 distances structures successfully saved to UPDATE_WITH_YOUR_PATH/output/binary/sentiment/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/eval/semeval2017/exemplar/db_orig-train_balanced_test_dist_dir
# Loading exemplar database data for analysis.
# Total number of database tokens: 277720
# Exemplar database data loaded.
# Loading original database sentences and labels.
# Original database sentences and labels loaded.
# Loading exemplar eval data.
# Total number of query tokens: 74834
# Exemplar data loaded.
# Creating non-padding masks for 4750 sentences.

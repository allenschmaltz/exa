################################################################################
#### uniCNN+BERT -- eval the K-NN model on the semeval2017 test set
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
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/semeval2017"
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

################# query
# semeval2017 data:
SPLIT_NAME="test"
DIR_PREFIX="balanced"
DATA_INPUT_NAME="SemEval2017-task4-${SPLIT_NAME}.subtask-A.english.txt.binaryevalformat.${DIR_PREFIX}.txt"

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/SemEval2017-task4-test/binaryevalformat"


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/semeval2017/exemplar/db_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}_${DIR_PREFIX}_${SPLIT_NAME}_dist_dir"



TOP_K=8

MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"


LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/binary/sentiment/${EXPERIMENT_LABEL}/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"



DATABASE_SPLIT_NAME=""  # this is blank for these experiments---always train orig "${DB_SPLIT_NAME}_${TRAINING_DIR_PREFIX}"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/semeval2017/dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}/"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/"${DATA_INPUT_NAME}" \
--test_seq_labels_file ${DATA_DIR}/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.monolabels.balanced.txt \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--epoch 60 \
--gpu 0 \
--score_vals_file "${OUTPUT_DIR}"/sentence_level_score_vals.dataprefix_${DIR_PREFIX}.split_${SPLIT_NAME}.maxdevepoch.zerorun.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot.viz.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.detection.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--dropout_probability ${DROPOUT_PROB} \
--exemplar_sentences_database_file ${DB_DATA_DIR}/${DB_TXT_FILE} \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/exemplar.dataprefix_${TRAINING_DIR_PREFIX}.${DB_SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_data_query_file "${SECOND_EXEMPLAR_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 10 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_dir="${LINEAR_EXA_MODEL_DIR}" \
--learning_rate 1.0  \
--approximation_type ${APPROX_TYPE} \
--top_k ${TOP_K} \
--saved_model_file "${LINEAR_EXA_MODEL_DIR}/aesw_non-static_best_max_sign_flips_to_original_model_epoch.pt" \
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--print_error_analysis \
--max_exemplars_to_return ${TOP_K} \
--output_analysis_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.savetype3.r1.txt \
--output_save_type 3 \
--binomial_sample_p 0.01 \
--output_annotations_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.annotations.r1.txt \
--save_annotations \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--save_prediction_stats >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.r1.log.txt

# Note that in these cases, the sequence labels are mono labels based on the class, only for debugging.

# The output file --output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt (linear_exa.epoch31.database_.query_test.tanh_knn8_fce_dev_split_temp10.0_model-maxent.eval.output_archive.r1.pt) is subsequently used below.


################################################################################
#### analyze constraints with K-NN on out-of-domain data at the review-level
#### The constraints are derived from the in-domain dev set (see sentiment_data/model/orig3k/orig3k_model__linear_exa_train_eval.sh)
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

# semeval2017:
OUTPUT_DIR=UPDATE_WITH_YOUR_PATH/output/zero_shot_labeling/fce_cnn_interp/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/semeval2017/dataprefix_orig-train.dataprefix_balanced.database_.query_test
SPLIT_NAME="test"

#SPLIT_NAME="train"
DATABASE_SPLIT_NAME=""
#DATABASE_SPLIT_NAME="dev"

EPOCH="31"

TOP_K=8

MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

python exa_analysis_sentence_level_analysis.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "KNN" \
--constrain_by_nearest_distance \
--class0_distance_threshold 34.325656214097 \
--class1_distance_threshold 35.43969287099065 \
--constrain_by_knn_output_magnitude \
--class0_magnitude_threshold 0.12846459534990023 \
--class1_magnitude_threshold 0.214461833364285 \
--admitted_tokens_proportion_min 0.10126582278481013 \
--admitted_tokens_total_min 5.068219055358903 \
--admitted_tokens_total_max 15.416629429489582

    # Sentence-level Accuracy: 0.7781052631578947 out of 4750 sentences
    # Proportion of ground-truth class 1: 0.5
    # Proportion of predicted class 1: 0.42378947368421055
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 1.0 out of 1 sentences
    # Proportion of ground-truth class 1 among constrained: 1.0
    # Proportion of predicted class 1 among constrained: 1.0


# additionally, we consider only using the proportion floor:

python exa_analysis_sentence_level_analysis.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "KNN" \
--constrain_by_nearest_distance \
--class0_distance_threshold 34.325656214097 \
--class1_distance_threshold 35.43969287099065 \
--constrain_by_knn_output_magnitude \
--class0_magnitude_threshold 0.12846459534990023 \
--class1_magnitude_threshold 0.214461833364285 \
--admitted_tokens_proportion_min 0.10126582278481013


    # Sentence-level Accuracy: 0.7781052631578947 out of 4750 sentences
    # Proportion of ground-truth class 1: 0.5
    # Proportion of predicted class 1: 0.42378947368421055
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 0.8142361111111112 out of 576 sentences
    # Proportion of ground-truth class 1 among constrained: 0.4045138888888889
    # Proportion of predicted class 1 among constrained: 0.2916666666666667

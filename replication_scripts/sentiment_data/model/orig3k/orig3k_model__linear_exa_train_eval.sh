################################################################################
#### uniCNN+BERT -- train a K-NN model
#### Note that for the standard zero shot setting, the model does not have
#### access to token labels from training, but does have sentence level labels
#### from training.
#### Training the K-NN model uses 1/2 the dev set, with the other half
#### serving as the K-NN dev set. These splits must be created before training.
#### Note that we use --input_is_untokenized with the sentiment experiments.
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
# for output on second drive:
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
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

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}k.txt \
--epoch 60 \
--gpu 0 \
--saved_model_file "" \
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
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--model_temperature_init_value ${INIT_TEMP} \
--model_support_weights_init_values="1.0,-1.0,2.0" \
--model_bias_init_value="0.0" \
--model_gamma_init_value="1.0" \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--query_train_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-train-split_chunks_ids.pt" \
--query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
--restrict_eval_to_query_eval_split_chunk_ids_file >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.train.r1.log.txt



######### MODEL_TYPE="maxent" #########

# TOP_K=8
#
# MODEL_TYPE="maxent"
# INIT_TEMP=10.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
# EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"
#
# Total sign flips (relative to original model): 2901 out of 14447; As percent accuracy: 0.7991970651346301
# Total sign flips (relative to ground-truth): 3574 out of 14447; As percent accuracy: 0.7526129992385963
#
# model.model_bias: -0.029825497418642044
# model.model_gamma (y_n): 0.003454119898378849
# model.model_temperature: 10.015738487243652
#
# Current max dev sign_flips_to_original_model score: 2901 at epoch 45


################################################################################
#### uniCNN+BERT -- eval the K-NN model from above on the dev set
#### The training script will output the eval results on the K-NN dev set split,
#### but if we need to re-run, or print out output for analysis, we can
#### reproduce them as here. Note that this differs from the test script below
#### in that we use
#### --query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
#### --restrict_eval_to_query_eval_split_chunk_ids_file
#### (The section after this shows how to calculate the cutoffs for the heuristics for
#### abstaining from predicting over domain-shifted/out-of-domain data (at the *review level*) using the
#### *token-level* thresholds determined here (against the approximations---i.e., without
#### accessing the token-level sequence labels). We use the full dev set in the next
#### section to relate these token-level thresholds to the review-level predictions, using
#### the full dev set, rather than just the K-NN dev, since the latter
#### is particularly small in this case---only ~245/2 reviews.)
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
# for output on second drive:
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
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


# changing output dir to avoid clobbering the output files:
DATABASE_SPLIT_NAME=""  # this is blank for these experiments---always train orig "${DB_SPLIT_NAME}_${TRAINING_DIR_PREFIX}"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}/"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
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
--query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
--restrict_eval_to_query_eval_split_chunk_ids_file \
--save_prediction_stats >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval_knndevsplit.r1.log.txt

# additional analysis:
python exa_analysis_quantiles_approximations.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--format_output_for_paper \
--show_combined_magnitude_direction_quantiles >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval_knndevsplit.exa_analysis_quantiles_approximations.log.txt


python exa_analysis_quantiles.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--format_output_for_paper \
--show_combined_magnitude_direction_quantiles

# We include the output to this commented below. This produces the constraint values used in the sections below.
python exa_analysis_summary_stats.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--output_metrics_against_ground_truth >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.exa_analysis_summary_stats.log.txt

# ######### MODEL_TYPE="maxent" #########
#
# TOP_K=8
#
# MODEL_TYPE="maxent"
# INIT_TEMP=10.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
# EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

# --------Approximation is correct: True--------
# *Class: 0*
# K-NN output:
#         Label: knn_logit_approximationTrue_class0
#                 mean: -0.12846459534990023; min: -1.0332796573638916; max: -1.7564743757247925e-06; std: 0.1981862636976113; total: 9486
# K-NN distance:
#         Label: knn_logit_approximationTrue_class0_distance
#                 mean: 34.325656214097; min: 5.550702095031738; max: 60.67125701904297; std: 9.620989251483293; total: 5836
# *Class: 1*
# K-NN output:
#         Label: knn_logit_approximationTrue_class1
#                 mean: 0.214461833364285; min: 4.999339580535889e-06; max: 0.973588764667511; std: 0.21838459568409538; total: 2060
# K-NN distance:
#         Label: knn_logit_approximationTrue_class1_distance
#                 mean: 35.43969287099065; min: 11.677870750427246; max: 54.819419860839844; std: 8.182738277746331; total: 1332
# *Class: 0 and 1*
# K-NN distance (both classes combined):
#         Label: knn_logit_approximationTrue_class0AND1_distance
#                 mean: 34.53267307053985; min: 5.550702095031738; max: 60.67125701904297; std: 9.38045320331656; total: 7168
# *Class: 0*
# Original Model output:
#         Label: original_model_logit_approximationTrue_class0
#                 mean: -0.28636985641887447; min: -37.55717849731445; max: -5.1895622164011e-05; std: 1.3561087469955568; total: 9486
# Original Model distance:
#         Label: original_model_logit_approximationTrue_class0_distance
#                 mean: 34.325656214097; min: 5.550702095031738; max: 60.67125701904297; std: 9.620989251483293; total: 5836
# *Class: 1*
# Original Model output:
#         Label: original_model_logit_approximationTrue_class1
#                 mean: 0.6771204168416919; min: 1.2326519936323166e-05; max: 15.80837345123291; std: 1.1706983647783489; total: 2060
# Original Model distance:
#         Label: original_model_logit_approximationTrue_class1_distance
#                 mean: 35.43969287099065; min: 11.677870750427246; max: 54.819419860839844; std: 8.182738277746331; total: 1332
# *Class: 0 and 1*
# Original Model distance (both classes combined):
#         Label: original_model_logit_approximationTrue_class0AND1_distance
#                 mean: 34.53267307053985; min: 5.550702095031738; max: 60.67125701904297; std: 9.38045320331656; total: 7168


################################################################################
#### uniCNN+BERT -- eval the K-NN model from above on the test sets
#### This is for the orig and new test sets.
#### Separately here, we also run on orig dev for analysis, which is the full dev instead of
#### the ~50% split from above. This full dev set is used to tune the sentence-level constraints
#### (i.e., the threshold on proportion of admitted tokens, w.r.t. to the
#### sentence-level label).
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
# for output on second drive:
SECOND_EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
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
### USE EITHER DIR_PREFIX="orig" (for the original data) or DIR_PREFIX="new" for the counter-factually re-written, commenting the other
DIR_PREFIX="orig"
#DIR_PREFIX="new" # when eval in other directory, rewrite above (which is used for model dir)
EVAL_DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/${DIR_PREFIX}/binaryevalformat/aligned_sequence_labels"


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/binary/sentiment/${EXPERIMENT_LABEL}/eval/aligned/exemplar/db_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}_${DIR_PREFIX}_${SPLIT_NAME}_dist_dir"


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


# changing output dir to avoid clobbering the output files:
if [ ${SPLIT_NAME} == "dev" ]; then
  DATABASE_SPLIT_NAME=""  # this is blank for these experiments---always train orig "${DB_SPLIT_NAME}_${TRAINING_DIR_PREFIX}"
  # we add an extra prefix here ("_alldev") to avoid clobbering the smaller knn dev split output from the previous section
  OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}_alldev/"
  OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs
else
  DATABASE_SPLIT_NAME=""  # this is blank for these experiments---always train orig "${DB_SPLIT_NAME}_${TRAINING_DIR_PREFIX}"
  OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_${TRAINING_DIR_PREFIX}-${DB_SPLIT_NAME}.dataprefix_${DIR_PREFIX}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}/"
  OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs
fi

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${EVAL_DATA_DIR}/"${EVAL_DATA_INPUT_NAME}" \
--test_seq_labels_file ${EVAL_DATA_DIR}/${EVAL_SEQ_LABELS_INPUT_NAME} \
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
--exemplar_data_query_file "${EXEMPLAR_DIR}"/exemplar.dataprefix_${DIR_PREFIX}.${SPLIT_NAME}.epoch${EPOCH}.exemplar_data.txt \
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


#### Evaluate using the inference-time decision rules:
# Note that this corresponds to the Orig.+ExAG (3.4k) row, columns 1 and 2, of Table 10 (arXiv v6)
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAG"

### on the original data:

# Number of sentences under consideration for analysis: 488
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.1574058716205009
#         Recall: 0.24074074074074073
#         F1: 0.19035202086049546
#         F0.5: 0.16911397818803908
#         MCC: 0.1422029670410084

### on the counterfactually re-written:

# Number of sentences under consideration for analysis: 488
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.15377730039384174
#         Recall: 0.1748066748066748
#         F1: 0.1636190476190476
#         F0.5: 0.1575684202802847
#         MCC: 0.10463379406950228

# Note that this corresponds to the Orig.+ExAT (3.4k) row, columns 1 and 2, of Table 10 (arXiv v6)
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAT"

### on the original data:

# Number of sentences under consideration for analysis: 488
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.518942731277533
#         Recall: 0.14941653982749872
#         F1: 0.23202678747291705
#         F0.5: 0.3472058476774345
#         MCC: 0.26017620921177015

### on the counterfactually re-written:

# Number of sentences under consideration for analysis: 488
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.5151515151515151
#         Recall: 0.10032560032560033
#         F1: 0.16794413217509793
#         F0.5: 0.2819720887668726
#         MCC: 0.2076877851014241



################################################################################
#### Analyze constraints with the K-NN on domain-shifted data at the review-level.
#### First, we get summary stats on the full orig dev set w.r.t. to the
#### review-level predictions. Note that the distance and magnitude thresholds
#### are determined above via exa_analysis_summary_stats.py. Here,
#### we determine cutoffs for the admitted tokens measured against the
#### review-level predictions.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

# # orig dev:
OUTPUT_DIR=UPDATE_WITH_YOUR_PATH/output/zero_shot_labeling/fce_cnn_interp/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_orig-train.dataprefix_orig.database_.query_dev_alldev
SPLIT_NAME="dev"

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


python exa_analysis_sentence_level_analysis_summary_stats.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "KNN" \
--constrain_by_nearest_distance \
--class0_distance_threshold 34.325656214097 \
--class1_distance_threshold 35.43969287099065 \
--constrain_by_knn_output_magnitude \
--class0_magnitude_threshold 0.12846459534990023 \
--class1_magnitude_threshold 0.214461833364285


    # Sentence-level Accuracy: 0.9428571428571428 out of 245 sentences
    # Proportion of ground-truth class 1: 0.5020408163265306
    # Proportion of predicted class 1: 0.4775510204081633
    # -------Note the following are restricted to sentences with at least 1 admitted token-------
    # Sentence-level Accuracy (Among admitted): 0.9428571428571428 out of 245 sentences
    # Proportion of ground-truth class 1 among constrained: 0.5020408163265306
    # Proportion of predicted class 1 among constrained: 0.4775510204081633
    # Among tokens in correct sentences, total admitted tokens: mean: 10.242424242424242; std: 5.174205187065339; min: 2: max: 28
    # Among tokens in correct sentences, proportion admitted: mean: 0.07098633272159904; std: 0.04197968061733813; min: 0.010869565217391304: max: 0.2604166666666667
    # Among tokens in wrong sentences, total admitted tokens: mean: 7.642857142857143; std: 2.8687265182890287; min: 2: max: 11
    # Among tokens in wrong sentences, proportion admitted: mean: 0.049781821952755434; std: 0.025593133349149245; min: 0.008583690987124463: max: 0.10126582278481013


# also, run constraints on the dev for reference:
#--admitted_tokens_proportion_min is the max for wrong predictions
#--admitted_tokens_total_min and --admitted_tokens_total_max are +/- 1 std. from mean for correct predictions:
# 10.242424242424242-5.174205187065339=5.068219055358903
# 10.242424242424242+5.174205187065339=15.416629429489582
# in effect 5 and 15 tokens

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

    # Sentence-level Accuracy: 0.9428571428571428 out of 245 sentences
    # Proportion of ground-truth class 1: 0.5020408163265306
    # Proportion of predicted class 1: 0.4775510204081633
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 1.0 out of 18 sentences
    # Proportion of ground-truth class 1 among constrained: 0.4444444444444444
    # Proportion of predicted class 1 among constrained: 0.4444444444444444

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

    # Sentence-level Accuracy: 0.9428571428571428 out of 245 sentences
    # Proportion of ground-truth class 1: 0.5020408163265306
    # Proportion of predicted class 1: 0.4775510204081633
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 1.0 out of 43 sentences
    # Proportion of ground-truth class 1 among constrained: 0.32558139534883723
    # Proportion of predicted class 1 among constrained: 0.32558139534883723

# Variations on this theme are possible now that we have these levers for
# measuring data and model uncertainty, and we could also aim for 'uncertainty quantification',
# where we define that phrase as methods that place the estimates in terms of a well-formed/calibrated
# probability estimate or bounds. (We might consider, for example, using conformal methods.)
# However, interestingly, what we show above is quite possibly what will
# make sense for many higher-risk settings in practicing. Namely, we set thresholds conservatively
# such that accuracy/effectiveness of admitted predictions on the known held-out set(s) is at the level
# of human annotation error.

################################################################################
#### Analyze constraints with K-NN on domain-shifted data at the review-level
#### on the original test set. Here, we use the constraints derived above on
#### the dev set and apply them to the test set.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


# orig
OUTPUT_DIR=UPDATE_WITH_YOUR_PATH/output/zero_shot_labeling/fce_cnn_interp/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_orig-train.dataprefix_orig.database_.query_test
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

    # Number of sentences under consideration for analysis: 488
    # Admitted but incorrect prediction sentence index: 8, true label: 0
    # Admitted but incorrect prediction sentence index: 250, true label: 1
    # Admitted but incorrect prediction sentence index: 264, true label: 1
    # -------------------------------------------------------
    #
    # Sentence-level Accuracy: 0.9282786885245902 out of 488 sentences
    # Proportion of ground-truth class 1: 0.5020491803278688
    # Proportion of predicted class 1: 0.47540983606557374
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 0.9302325581395349 out of 43 sentences
    # Proportion of ground-truth class 1 among constrained: 0.3953488372093023
    # Proportion of predicted class 1 among constrained: 0.37209302325581395

# # The review at index 8 appears to be mislabeled.
# UPDATE_WITH_YOUR_PATH/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/orig/binaryevalformat/aligned_sequence_labels/test.aligned_binaryevalformat.txt

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

    # Number of sentences under consideration for analysis: 488
    # Admitted but incorrect prediction sentence index: 8, true label: 0
    # Admitted but incorrect prediction sentence index: 250, true label: 1
    # Admitted but incorrect prediction sentence index: 264, true label: 1
    # -------------------------------------------------------
    # Sentence-level Accuracy: 0.9282786885245902 out of 488 sentences
    # Proportion of ground-truth class 1: 0.5020491803278688
    # Proportion of predicted class 1: 0.47540983606557374
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 0.9615384615384616 out of 78 sentences
    # Proportion of ground-truth class 1 among constrained: 0.358974358974359
    # Proportion of predicted class 1 among constrained: 0.34615384615384615


################################################################################
#### Analyze constraints with K-NN on domain-shifted data at the review-level
#### on the new (counterfactually revised) test set. Here, we use the constraints derived above on
#### the dev set and apply them to the test set.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

# new
OUTPUT_DIR=UPDATE_WITH_YOUR_PATH/output/zero_shot_labeling/fce_cnn_interp/counter_orig3k_unicnnbert_v7500k_bertlargecased_top4layers_fn1000_dropout0.50_maxlen350/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/dataprefix_orig-train.dataprefix_new.database_.query_test
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

    # Number of sentences under consideration for analysis: 488
    # Admitted but incorrect prediction sentence index: 16, true label: 1
    # -------------------------------------------------------
    # Sentence-level Accuracy: 0.8872950819672131 out of 488 sentences
    # Proportion of ground-truth class 1: 0.4979508196721312
    # Proportion of predicted class 1: 0.42213114754098363
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 0.9696969696969697 out of 33 sentences
    # Proportion of ground-truth class 1 among constrained: 0.42424242424242425
    # Proportion of predicted class 1 among constrained: 0.3939393939393939

# # The missed review is ambiguous.
# UPDATE_WITH_YOUR_PATH/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/new/binaryevalformat/aligned_sequence_labels/test.aligned_binaryevalformat.txt

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

    # Number of sentences under consideration for analysis: 488
    # Admitted but incorrect prediction sentence index: 16, true label: 1
    # -------------------------------------------------------
    # Sentence-level Accuracy: 0.8872950819672131 out of 488 sentences
    # Proportion of ground-truth class 1: 0.4979508196721312
    # Proportion of predicted class 1: 0.42213114754098363
    # -------Constrained sentences-------
    # Sentence-level Accuracy (Among admitted): 0.9807692307692307 out of 52 sentences
    # Proportion of ground-truth class 1 among constrained: 0.40384615384615385
    # Proportion of predicted class 1 among constrained: 0.38461538461538464

################################################################################
#### uniCNN+BERT -- train a K-NN model
#### Note that for the standard zero shot setting, the model does not have
#### access to token labels from training, but does have sentence level labels
#### from training.
#### Training the K-NN model uses 1/2 the FCE dev set, with the other half
#### serving as the dev set. These splits must be created before training.
#### You can uncomment the blocks for MODEL_TYPE="knn"
#### xor MODEL_TYPE="learned_weighting" to alternatively train the equally-weighted
#### K-NN or the constraint-weighted K-NN, respectively. The variable TOP_K
#### determines the number of nearest neighbors and INIT_TEMP
#### determines the initial value of the temperature parameter. On the datasets considered
#### in this work, we find that a relatively high initial temperature (resulting in a more diffuse
#### distribution for w_k) works well, and we start with INIT_TEMP=10.0 for all
#### KNN_dist. models across datasets and models. For KNN_const., we start with INIT_TEMP=5.0
#### and --model_support_weights_init_values="1.0,-1.0,2.0". For all models, the
#### bias is initialized at 0.0 and gamma at 1.0.
#### In this case we use --use_sentence_level_database_ground_truth, since
#### the support set only has access to document-level labels.
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="15"


#SPLIT_NAME="test"
SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/exemplar/${SPLIT_NAME}_dist_dir"
#mkdir -p "${DISTANCE_DEV_DIR}"

TOP_K=8
#TOP_K=1

# KNN_dist.
MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# # KNN_equal
# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# # KNN_const.
# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
--gpu 0 \
--saved_model_file "" \
--score_vals_file "${OUTPUT_DIR}"/zero_shot_ref_run.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--exemplar_sentences_database_file ${DATA_DIR}/fce-public.${DATABASE_SPLIT_NAME}.original.binaryevalformat.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
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
--restrict_eval_to_query_eval_split_chunk_ids_file >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.train.r1.log.txt


######### MODEL_TYPE="maxent" #########
# TOP_K=8
#
# MODEL_TYPE="maxent"
# INIT_TEMP=10.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
# EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

# model.model_bias: -0.21801362931728363
# model.model_gamma (y_n): 0.4125238060951233
# model.model_temperature: 10.024066925048828


################################################################################
#### uniCNN+BERT -- eval the K-NN model from above on the FCE dev set
#### The training script will output the eval results on the K-NN dev set split,
#### but if we need to re-run, or print out output for analysis, we can
#### reproduce them as here. Note that this differs from the test script below
#### in that we use
#### --query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
#### --restrict_eval_to_query_eval_split_chunk_ids_file
#### We additionally collect summary stats about the distances and output
#### magnitudes, which we use in the experiments to constrain the output.
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="15"


#SPLIT_NAME="test"
SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/exemplar/${SPLIT_NAME}_dist_dir"
# the database data structure is stored in the dev directory
DATABASE_DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/exemplar/dev_dist_dir"
#mkdir -p "${DISTANCE_DEV_DIR}"

TOP_K=8
#TOP_K=1
# TOP_K=25

MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
#
# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"


LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"


# changing output dir to scratch:
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
--gpu 0 \
--score_vals_file "${OUTPUT_DIR}"/zero_shot_ref_run.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--exemplar_sentences_database_file ${DATA_DIR}/fce-public.${DATABASE_SPLIT_NAME}.original.binaryevalformat.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_dir="${LINEAR_EXA_MODEL_DIR}" \
--learning_rate 1.0  \
--approximation_type ${APPROX_TYPE} \
--top_k ${TOP_K} \
--saved_model_file "${LINEAR_EXA_MODEL_DIR}/aesw_non-static_best_max_sign_flips_to_original_model_epoch.pt" \
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--database_data_structure_file "${DATABASE_DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
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
--save_prediction_stats >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval_knndevsplit.r1.log.txt

## additional analysis scripts:

# quantiles for approximations: \hat{y}_n==\hat{y}^{KNN}_n
python exa_analysis_quantiles_approximations.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--format_output_for_paper \
--show_combined_magnitude_direction_quantiles >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval_knndevsplit.exa_analysis_quantiles_approximations.log.txt

# additional quantile-based analysis
python exa_analysis_quantiles.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--format_output_for_paper \
--show_combined_magnitude_direction_quantiles

# get thresholds by class (distance and output magnitudes):
# These thresholds are then subsequently applied for inference over the
# FCE+news2k set (when NOT adding the news50k data to the support set--i.e.,
# it's otherwise the same model trained above).
# See fce_data/out_of_domain/base_model/base_model__eval_ood_google_data.sh.
python exa_analysis_summary_stats.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--output_metrics_against_ground_truth >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.exa_analysis_summary_stats.log.txt






################################################################################
#### uniCNN+BERT -- eval the K-NN model from above on the FCE test set
################################################################################


SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="15"


SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/exemplar/${SPLIT_NAME}_dist_dir"
# the database data structure is stored in the dev directory
DATABASE_DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/exemplar/dev_dist_dir"
#mkdir -p "${DISTANCE_DEV_DIR}"

TOP_K=8
# TOP_K=1


MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"


LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"


# changing output dir to scratch:
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
--gpu 0 \
--score_vals_file "${OUTPUT_DIR}"/zero_shot_ref_run.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--exemplar_sentences_database_file ${DATA_DIR}/fce-public.${DATABASE_SPLIT_NAME}.original.binaryevalformat.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_dir="${LINEAR_EXA_MODEL_DIR}" \
--learning_rate 1.0  \
--approximation_type ${APPROX_TYPE} \
--top_k ${TOP_K} \
--saved_model_file "${LINEAR_EXA_MODEL_DIR}/aesw_non-static_best_max_sign_flips_to_original_model_epoch.pt" \
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--database_data_structure_file "${DATABASE_DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--print_error_analysis \
--max_exemplars_to_return ${TOP_K} \
--output_analysis_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.savetype3.r1.txt \
--output_save_type 3 \
--binomial_sample_p 0.01 \
--output_annotations_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.annotations.r1.txt \
--save_annotations \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--save_prediction_stats >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.r1.log.txt

######### MODEL_TYPE="maxent" #########
# TOP_K=8
#
# MODEL_TYPE="maxent"
# INIT_TEMP=10.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
# EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"
#
# Total sign flips (relative to original model): 2638 out of 41159; As percent accuracy: 0.9359070920090381
# Total sign flips (relative to ground-truth): 6093 out of 41159; As percent accuracy: 0.8519643334386161
# Reference total sign flips (relative to ground-truth) of the *original model*: 6423 out of 41159; As percent accuracy: 0.8439466459340605
#
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.5118829981718465
#         Recall: 0.35527359238699446
#         F1: 0.4194363823612021
#         F0.5: 0.47041034902767864
#         MCC: 0.34423815587795853


# Evaluate using the inference-time decision rules:
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAG"

# Number of sentences under consideration for analysis: 2720
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.5678679690131357
#         Recall: 0.26740681998413957
#         F1: 0.3635971533318956
#         F0.5: 0.4636708651889334
#         MCC: 0.3216168982158497

python exa_analysis_quantiles.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "original_model" \
--format_output_for_paper



################################################################################
#### create graphs -- final K-NN dev
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

OUTPUT_DIR=UPDATE_WITH_YOUR_PATH/output/zero_shot_labeling/fce_cnn_interp/bert_cnn_v2_fce_v7500k_google_bertlargecased_top4layers_fn1000_fw1/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned
GRAPH_OUTPUT_DIR=${OUTPUT_DIR}/graph
mkdir -p ${GRAPH_OUTPUT_DIR}

#SPLIT_NAME="test"
SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

EPOCH="15"

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

CLASS_TO_RESTRICT=1
python exa_analysis_quantiles_approximations_graph.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "KNN" \
--format_output_for_paper \
--num_quantiles 5 \
--graph_type 0 \
--class_to_restrict ${CLASS_TO_RESTRICT} \
--output_graph_file "${GRAPH_OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.knn-dev.approximations.graph.class${CLASS_TO_RESTRICT}.base_f05.eps

cp "${GRAPH_OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.knn-dev.approximations.graph.class${CLASS_TO_RESTRICT}.base_f05.eps UPDATE_WITH_YOUR_PATH/knndev_approximation_base_class${CLASS_TO_RESTRICT}.eps

CLASS_TO_RESTRICT=0
python exa_analysis_quantiles_approximations_graph.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "KNN" \
--format_output_for_paper \
--num_quantiles 5 \
--graph_type 0 \
--class_to_restrict ${CLASS_TO_RESTRICT} \
--output_graph_file "${GRAPH_OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.knn-dev.approximations.graph.class${CLASS_TO_RESTRICT}.base_f05.eps

cp "${GRAPH_OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.knn-dev.approximations.graph.class${CLASS_TO_RESTRICT}.base_f05.eps UPDATE_WITH_YOUR_PATH/knndev_approximation_base_class${CLASS_TO_RESTRICT}.eps

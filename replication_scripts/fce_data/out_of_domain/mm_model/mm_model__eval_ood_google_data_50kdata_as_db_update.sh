################################################################################
#### MM loss: uniCNN+BERT  -- K-NN models trained only on the FCE training set,
#### here, evaluated on the out-of-domain augmented set WITH the 50k Google
#### sentences added to the database.
#### Note that for the standard zero shot setting, the model does not have
#### access to token labels from training, but does have sentence level labels
#### Note that we need the update flag (--update_knn_model_database) in order
#### to update the database with the extended set.
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
COMBINED_DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/corpora/lm/billion/processed/for_decorrelater_experiments/cat_with_fce
ADD_DATA_SIZE=50000
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_cnn_v2_fce_v${VOCAB_SIZE}k_google_bertlargecased_top4layers_fn${FILTER_NUMS}_fw1

MODEL_DIR=${SERVER_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15"
EXEMPLAR_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/out/exemplar"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/out/exemplar/eval/revision_experiments/linearexa/standard_fcetrain_with50kaugmented_db"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="14"  # best in regard to sentence F1 from the fully-connected layer

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

DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/exemplar/${SPLIT_NAME}_dist_dir/google2k_with50kdb"
DATABASE_DISTANCE_DEV_DIR="${DISTANCE_DEV_DIR}"
#mkdir -p "${DISTANCE_DEV_DIR}"

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


LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"

# changing output dir to sratch:
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/out/exemplar/eval/revision_experiments/linearexa_tanh_nearest_options_learned/standard_fcetrain_with50kaugmented_db"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

ADD_EXPERIMENT_ID="google50k_augmented_db"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_size2000.txt \
--test_seq_labels_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_labels_size2000.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
--gpu 0 \
--score_vals_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.zero_shot_ref_run.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.zero_shot_ref_run.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.zero_shot_ref_run.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--exemplar_sentences_database_file ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_train_size${ADD_DATA_SIZE}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}/fcetrain_with_google${ADD_DATA_SIZE}"/zero_shot.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/fce_with_google/zero_shot.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.exemplar.txt \
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
--database_data_structure_file "${DATABASE_DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--print_error_analysis \
--max_exemplars_to_return ${TOP_K} \
--output_analysis_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.savetype3.r1.txt \
--output_save_type 3 \
--binomial_sample_p 0.01 \
--output_annotations_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.annotations.r1b.txt \
--save_annotations \
--output_prediction_stats_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--save_prediction_stats \
--update_knn_model_database >"${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.r1.log.txt



# Evaluate using the inference-time decision rules:
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAG"

# Number of sentences under consideration for analysis: 4720
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.6508494434680726
#         Recall: 0.17620935765265663
#         F1: 0.27733399900149774
#         F0.5: 0.42298027868727633
#         MCC: 0.3170418304636864

# This is the case where the inference-time decision rule has access to *token-level* labels.
# This provides a contrast/comparison to training with the token-level labels.
# Note the increase in precision relative to ExAG.
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/${ADD_EXPERIMENT_ID}.linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAT"

# Number of sentences under consideration for analysis: 4720
# -------------------------------------------------------
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.7815699658703071
#         Recall: 0.14528152260111024
#         F1: 0.24501805536980073
#         F0.5: 0.41662876375875557
#         MCC: 0.320687288710735

# ######### MODEL_TYPE="maxent" #########
#
# TOP_K=8
# MODEL_TYPE="maxent"
# INIT_TEMP=10.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
# EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"
#
#
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.6448422044276967
#         Recall: 0.21712926249008724
#         F1: 0.32486948267679167
#         F0.5: 0.4625937690072312
#         MCC: 0.35072836975576016

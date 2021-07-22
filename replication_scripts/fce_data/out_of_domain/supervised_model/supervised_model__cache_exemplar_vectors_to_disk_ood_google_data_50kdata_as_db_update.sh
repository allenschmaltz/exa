################################################################################
#### fully supervised: uniCNN+BERT+S*; cache exemplar distances to disk
#### Note that this model is only trained with the FCE dataset, but here the
#### Google News data gets added to the database.
################################################################################

################################################################################
#### fully supervised: uniCNN+BERT+S*; generate exemplar data
#### First we cache the exemplar vectors to disk as text files. In this case,
#### this is the new training set which consists of the original FCE
#### training set augmented with 50k already correct sentences. This file
#### will be around 20 GB.
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

FINE_TUNE_MODE_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}/exemplar/fcetrain_with_google${ADD_DATA_SIZE}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="18"


#SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

# only training; we process the modified dev set separately
for SPLIT_NAME in "train";
do

  CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
  --mode "generate_exemplar_data" \
  --model "non-static" \
  --dataset "aesw" \
  --word_embeddings_file ${MODEL_DIR}/not_used.txt \
  --test_file ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_train_size${ADD_DATA_SIZE}.txt \
  --test_seq_labels_file ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_train_labels_size${ADD_DATA_SIZE}.txt \
  --max_length 50 \
  --max_vocab_size 7500 \
  --vocab_file ${MODEL_DIR}/vocab7500k.txt \
  --epoch 20 \
  --learning_rate 1.0 \
  --gpu 0 \
  --saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
  --score_vals_file "${OUTPUT_DIR}"/zero_shot.score_vals.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.txt \
  --bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
  --bert_model ${BERT_MODEL} \
  --bert_layers=${NUM_LAYERS} \
  --bert_gpu 0 \
  --color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
  --visualization_out_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.viz.html \
  --correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
  --output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.detection.txt \
  --detection_offset 0 \
  --fce_eval \
  --forward_type ${FORWARD_TYPE} \
  --filter_widths="1" \
  --number_of_filter_maps="${FILTER_NUMS}" \
  --dropout_probability ${DROPOUT_PROB} \
  --output_exemplar_data_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.exemplar.txt

done



################################################################################
#### fully supervised: uniCNN+BERT+S*; generate exemplar data
#### First we cache the exemplar vectors to disk as text files. In this case,
#### this is the new test set which consists of the original FCE
#### test set augmented with 2k already correct sentences.
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

FINE_TUNE_MODE_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}/exemplar/fce_with_google"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="18"


#SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50


for SPLIT_NAME in "test";
do

  CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
  --mode "generate_exemplar_data" \
  --model "non-static" \
  --dataset "aesw" \
  --word_embeddings_file ${MODEL_DIR}/not_used.txt \
  --test_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_size2000.txt \
  --test_seq_labels_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_labels_size2000.txt \
  --max_length 50 \
  --max_vocab_size 7500 \
  --vocab_file ${MODEL_DIR}/vocab7500k.txt \
  --epoch 20 \
  --learning_rate 1.0 \
  --gpu 0 \
  --saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
  --score_vals_file "${OUTPUT_DIR}"/zero_shot.score_vals.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.txt \
  --bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
  --bert_model ${BERT_MODEL} \
  --bert_layers=${NUM_LAYERS} \
  --bert_gpu 0 \
  --color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
  --visualization_out_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.viz.html \
  --correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
  --output_generated_detection_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.detection.txt \
  --detection_offset 0 \
  --fce_eval \
  --forward_type ${FORWARD_TYPE} \
  --filter_widths="1" \
  --number_of_filter_maps="${FILTER_NUMS}" \
  --dropout_probability ${DROPOUT_PROB} \
  --output_exemplar_data_file "${OUTPUT_DIR}"/zero_shot.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.exemplar.txt

done


################################################################################
#### fully supervised: uniCNN+BERT+S*; -- save exemplar distance structures
#### This model is trained using the standard database (i.e., only fce training)
#### Here we calculate the Google 2k+FCE test set distances using the train
#### set with 50k sentences from Google 1 Billion dataset added to the
#### FCE set. Note that we need to save the database data structure,
#### since we are no longer using the FCE only set. We do this by
#### specifying --save_database_data_structure and
#### --database_data_structure_file
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

FINE_TUNE_MODE_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}"
# note that we've changed to the SERVER_SCRATCH_DRIVE_PATH_PREFIX
EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}/exemplar/eval/revision_experiments/linearexa/standard_fcetrain_with50kaugmented_db"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="18"

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


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}/exemplar/${SPLIT_NAME}_dist_dir/google2k_with50kdb"
mkdir -p "${DISTANCE_DEV_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_size2000.txt \
--test_seq_labels_file ${COMBINED_DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt_with_google_1b_combined.binaryevalformat_${SPLIT_NAME}_labels_size2000.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
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
--exemplar_sentences_database_file ${COMBINED_DATA_DIR}/fce-public.train.original.binaryevalformat.txt_with_google_1b_combined.binaryevalformat_train_size${ADD_DATA_SIZE}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}/fcetrain_with_google${ADD_DATA_SIZE}"/zero_shot.epoch${EPOCH}.fcetrain_with_google${ADD_DATA_SIZE}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/fce_with_google/zero_shot.epoch${EPOCH}.fce_with_google_${SPLIT_NAME}.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 10 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_database_data_structure \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt

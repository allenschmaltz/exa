################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm; cache exemplar vectors and
#### distances to disk
#### Note that this model is trained with the conll2010uncertainty dataset.
################################################################################

################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm; generate exemplar data
#### First we cache the exemplar vectors to disk as text files. In a
#### subsequent step, we'll additionally cache all the distances to speed
#### up training the K-NN models.
#### Note this needs to be repeated three times, for each of the splits:
#SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"
SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_conll2010_v${VOCAB_SIZE}k_bertbaseuncased_top4layers_fn${FILTER_NUMS}_document_level

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="19"  # best in regard to sentence F1 from the fully-connected layer (i.e., this is the same epoch/model used for the zero-shot results)

#SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

for SPLIT_NAME in "test" "dev" "train";
do
  echo "Processing ${SPLIT_NAME}"

  CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
  --mode "generate_exemplar_data" \
  --model "non-static" \
  --dataset "aesw" \
  --word_embeddings_file ${MODEL_DIR}/not_used.txt \
  --test_file ${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt \
  --test_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt \
  --max_length 50 \
  --max_vocab_size 7500 \
  --vocab_file ${MODEL_DIR}/vocab7500k.txt \
  --epoch 20 \
  --learning_rate 1.0 \
  --gpu 0 \
  --saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
  --score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
  --bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
  --bert_model ${BERT_MODEL} \
  --bert_layers=${NUM_LAYERS} \
  --bert_gpu 0 \
  --color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
  --visualization_out_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
  --correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
  --output_generated_detection_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
  --detection_offset 0 \
  --fce_eval \
  --forward_type ${FORWARD_TYPE} \
  --filter_widths="1" \
  --number_of_filter_maps="${FILTER_NUMS}" \
  --dropout_probability ${DROPOUT_PROB} \
  --input_is_untokenized \
  --data_formatter "fce" \
  --do_lower_case \
  --output_exemplar_data_file "${OUTPUT_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.exemplar.txt
done


################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm; -- save exemplar distance structures
#### This is the dev set from the conll2010uncertainty set, which is then split
#### itself into 2 sets for training and dev of the K-NN.
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"
SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_conll2010_v${VOCAB_SIZE}k_bertbaseuncased_top4layers_fn${FILTER_NUMS}_document_level

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/linearexa"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="19"


#SPLIT_NAME="test"
SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/${SPLIT_NAME}_dist_dir"
mkdir -p "${DISTANCE_DEV_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt \
--test_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case \
--exemplar_sentences_database_file ${DATA_DIR}/conll2010uncertainty.${DATABASE_SPLIT_NAME}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_database_data_structure \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--create_train_eval_split_from_query_for_knn \
--binomial_sample_p 0.5 \
--query_train_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-train-split_chunks_ids.pt" \
--query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt



#SPLIT_NAME="dev"

    # Total number of database tokens: 390927
    # Exemplar database data loaded.
    # Loading original database sentences and labels.
    # Original database sentences and labels loaded.
    # Loading exemplar eval data.
    # Total number of query tokens: 48227
    # Exemplar data loaded.
    # Creating non-padding masks for 1960 sentences.
    # Number of chunks in the query-train split: 23
    # Number of chunks in the query-eval split: 17



################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm; -- save exemplar distance structures
#### Similarly, here we save the conll2010uncertainty test, but note here we do not need to
#### save a query-train/query-eval split.
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"
SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_conll2010_v${VOCAB_SIZE}k_bertbaseuncased_top4layers_fn${FILTER_NUMS}_document_level

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/linearexa"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="19"


SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/${SPLIT_NAME}_dist_dir"
mkdir -p "${DISTANCE_DEV_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "save_exemplar_distances" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt \
--test_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case \
--exemplar_sentences_database_file ${DATA_DIR}/conll2010uncertainty.${DATABASE_SPLIT_NAME}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--top_k 25 >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.log.r1.txt



SPLIT_NAME="test"


# Total number of query tokens: 47614
# Exemplar data loaded.
# Creating non-padding masks for 1937 sentences.

################################################################################
#### uniCNN+BERT_{base_uncased}+S*: uniCNN+BERT_{base_uncased} fine-tuned with supervised labels
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

FINE_TUNE_MODE_DIR="${MODEL_DIR}/seq_labeling_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/seq_labeling_fine_tune"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="10"  # this is for the original model; best in regard to sentence F1
#SPLIT_NAME="test"
FINE_TUNE_SPLIT_NAME="train"
SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "seq_labeling_fine_tune" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--training_file ${DATA_DIR}/conll2010uncertainty.train.txt \
--dev_file ${DATA_DIR}/conll2010uncertainty.dev.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--save_model \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${FINE_TUNE_MODE_DIR}" \
--saved_model_file ${MODEL_DIR}/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/seq_labeling_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/seq_labeling_fine_tune.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/seq_labeling_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--training_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.train.txt \
--dev_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.dev.txt \
--forward_type 1 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case \
--max_metric_label "fscore_0_5" >"${OUTPUT_LOG_DIR}"/conll2010.seq_labeling_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.log.txt


#Max dev fscore_0_5: 0.9015525758645024 at epoch 19


################################################################################
#### uniCNN+BERT_{base_uncased}+S*; For reference, we get the sentence-level scores by
#### the token-level logits. See Footnote 5 of the Online Appendix.
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

FINE_TUNE_MODE_DIR="${MODEL_DIR}/seq_labeling_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/seq_labeling_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="19"  # best in regard to sentence F0.5 token-level

SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test_by_contributions" \
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
--score_vals_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.test_by_contributions.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.test_by_contributions.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.test_by_contributions.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case



SPLIT_NAME="test"

# (Accuracy from the global fc (primarily for debugging purposes): 0.783686112545173)
# test acc: 0.9530201342281879

# get the F1 score
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_for_minmax.py \
--input_score_vals_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.test_by_contributions.txt \
--start_offset 0.0 \
--end_offset 0.0 \
--number_of_offsets 1


      # Percent of gold sentences with errors: 0.21631388745482705 (419 out of 1937)
      # ------------------------------------------------------------
      # RANDOM
      #         Predicted error count: 989
      #         Precision: 0.22143579373104147
      #         Recall: 0.522673031026253
      #         F1: 0.31107954545454547
      #         F0.5: 0.25028571428571433
      #         MCC: 0.012706101720049339
      # Accuracy: 0.49922560660815696
      # ------------------------------------------------------------
      # MAJORITY CLASS
      #         Predicted error count: 1937
      #         Precision: 0.21631388745482705
      #         Recall: 1.0
      #         F1: 0.35568760611205436
      #         F0.5: 0.25652014203501894
      #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
      #         MCC: 0.0
      # Accuracy: 0.21631388745482705
      # ------------------------------------------------------------
      # Offset: 0.0
      #         Predicted error count: 412
      #         Precision: 0.8980582524271845
      #         Recall: 0.883054892601432
      #         F1: 0.8904933814681106
      #         F0.5: 0.8950169327527818
      #         MCC: 0.8606408711875754
      # Accuracy: 0.9530201342281879



################################################################################
#### uniCNN+BERT_{base_uncased}+S*;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. The calculations are the same as in the zero-shot setting,
#### but here we are performing inference for the fully-supverised
#### sequence labeling setting.
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

FINE_TUNE_MODE_DIR="${MODEL_DIR}/seq_labeling_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/seq_labeling_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="19"  # best in regard to sentence F0.5 token-level

SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero" \
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
--score_vals_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval.seq_labeling_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case

# RANDOM CLASS
#         Precision: 0.012184873949579832
#         Recall: 0.4906937394247039
#         F1: 0.02377926284285187
#         F0.5: 0.015137121441471538
#         MCC: -0.0020536586749900216
#
# MAJORITY CLASS
#         Precision: 0.012412315705464779
#         Recall: 1.0
#         F1: 0.024520277979462712
#         F0.5: 0.015467398074819285
#         MCC: 0.0
#
#
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.907258064516129
#         Recall: 0.7614213197969543
#         F1: 0.827966881324747
#         F0.5: 0.8737864077669903
#         MCC: 0.829242121771828

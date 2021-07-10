#### Ablation run -- for analysis
################################################################################
#### uniCNN_{M=2}+BERT_base: Train the initial model using sentence-level labels
#### This uses BERT_base to compare to the recent approach using
#### weighted attention over a Transformer with a BERT_base parameter count.
#### *This model is purposefully weak to illustrate the
#### impact of the capacity of the filters. FILTER_NUMS=2*
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=2
EXPERIMENT_LABEL=bert_unicnn_fce_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}


OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${SERVER_DRIVE_PATH_PREFIX}/data/general/GoogleNews-vectors-negative300.bin \
--training_file ${DATA_DIR}/fce-public.train.original.binaryevalformat.txt \
--dev_file ${DATA_DIR}/fce-public.dev.original.binaryevalformat.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--save_model \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${MODEL_DIR}" \
--score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" >"${OUTPUT_LOG_DIR}"/train.fce_document_level.log.txt


#Max dev accuracy: 0.8285328532853286 at epoch 6

# At the moment, sentence-level tuning is not implemented in the main code,
# but we can just save all of the epochs and then run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20


# Max F1: 0.8695955369595536; epoch 6 offset 0.061224489795918366
# Max F0.5: 0.8773437499999999; epoch 11 offset 0.0
# Max MCC: 0.6342664786292195; epoch 6 offset 0.04081632653061224

# Note that this is tuning at the *sentence-level*; token-level labels have
# not yet been considered at this point.
# In the paper, we used the F1 epoch as the epoch for zero-shot labeling and
# then subsequent min-max fine-tuning.


################################################################################
#######  TEST (sentence level) uniCNN_{M=2}+BERT_base
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the test FCE test set.
################################################################################


SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=2
EXPERIMENT_LABEL=bert_unicnn_fce_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="6"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/fce-public.test.original.binaryevalformat.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/fce.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" >"${OUTPUT_LOG_DIR}"/fce.test.sentence_level_score_vals.log.txt


    # (Accuracy from random prediction (only for debugging purposes): 0.47463235294117645)
    # (Accuracy from all 1's prediction (only for debugging purposes): 0.6790441176470589)
    # Ground-truth Stats: Number of instances with class 1: 1847 out of 2720
    # test acc: 0.8121323529411765

# To get the particular value for the decision boundary determined above,
# run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/fce.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.061224489795918366 \
--end_offset 0.061224489795918366 \
--number_of_offsets 1

# Offset: 0.061224489795918366
#         Predicted error count: 1855
#         Precision: 0.8603773584905661
#         Recall: 0.8641039523551706
#         F1: 0.8622366288492707
#         F0.5: 0.861120103593396
#         MCC: 0.5688131934519085


################################################################################
#### uniCNN_{M=2}+BERT_base;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting, producing results
#### on the test split desired.
#### Note that by default for reference/analysis, we show the results for
#### varying the decision boundary, but the real-world scenario is the result
#### corresponding to a decision boundary of 0.
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=2
EXPERIMENT_LABEL=bert_unicnn_fce_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}


OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="6"

#### Choose the dev or test split, as desired -- typically for the zero-shot setting, we're only interested in test
SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--test_seq_labels_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/eval_binary_train.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval_binary_train.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval_binary_train.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}"


# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.5790973871733966
#         Recall: 0.19333862014274386
#         F1: 0.2898929845422117
#         F0.5: 0.4139219015280135
#         MCC: 0.27505106146153013

#### Additional comparison run -- for analysis
################################################################################
#### uniCNN+BERT_base: Train the initial model using sentence-level labels
#### This uses BERT_base to compare to the recent approach using
#### weighted attention over a Transformer with a BERT_base parameter count.
################################################################################

SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/fce_formatted
VOCAB_SIZE=7500
FILTER_NUMS=1000
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


#Max dev accuracy: 0.8312331233123312 at epoch 19

# At the moment, sentence-level tuning is not implemented in the main code,
# but we can just save all of the epochs and then run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

    # Percent of gold sentences with errors: 0.6372637263726373 (1416 out of 2222)
    # ------------------------------------------------------------
    # RANDOM
    #         Predicted error count: 1140
    #         Precision: 0.6131578947368421
    #         Recall: 0.4936440677966102
    #         F1: 0.5469483568075117
    #         F0.5: 0.5848393574297189
    #         MCC: -0.05146427052203981
    # ------------------------------------------------------------
    # MAJORITY CLASS
    #         Predicted error count: 2222
    #         Precision: 0.6372637263726373
    #         Recall: 1.0
    #         F1: 0.7784496976360638
    #         F0.5: 0.687111801242236
    #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
    #         MCC: 0.0

    # Max F1: 0.8750867453157529; epoch 15 offset 0.4693877551020408
    # Max F0.5: 0.8791936433195996; epoch 15 offset 0.08163265306122448
    # Max MCC: 0.6513330112953223; epoch 15 offset 0.2857142857142857


# Note that this is tuning at the *sentence-level*; token-level labels have
# not yet been considered at this point.
# In the paper, we used the F1 epoch as the epoch for zero-shot labeling and
# then subsequent min-max fine-tuning.


################################################################################
#######  TEST (sentence level) uniCNN+BERT_base
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
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_fce_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"

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


# (Accuracy from random prediction (only for debugging purposes): 0.4860294117647059)
# (Accuracy from all 1's prediction (only for debugging purposes): 0.6790441176470589)
# Ground-truth Stats: Number of instances with class 1: 1847 out of 2720
# test acc: 0.8011029411764706

# To get the particular value for the decision boundary determined above,
# run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/fce.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.4693877551020408 \
--end_offset 0.4693877551020408 \
--number_of_offsets 1

      # Percent of gold sentences with errors: 0.6790441176470589 (1847 out of 2720)
      # ------------------------------------------------------------
      # RANDOM
      #         Predicted error count: 1395
      #         Precision: 0.6774193548387096
      #         Recall: 0.5116404981050352
      #         F1: 0.5829734731647132
      #         F0.5: 0.6361922714420358
      #         MCC: -0.003571062641642136
      # ------------------------------------------------------------
      # MAJORITY CLASS
      #         Predicted error count: 2720
      #         Precision: 0.6790441176470589
      #         Recall: 1.0
      #         F1: 0.808846069629954
      #         F0.5: 0.7256226919148269
      #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
      #         MCC: 0.0
      # ------------------------------------------------------------
      # Offset: 0.4693877551020408
      #         Predicted error count: 1843
      #         Precision: 0.8638090070537168
      #         Recall: 0.8619382782891175
      #         F1: 0.8628726287262871
      #         F0.5: 0.8634342119535741
      #         MCC: 0.5737340851491873


################################################################################
#### uniCNN+BERT_base;
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
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_fce_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}


OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"

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



# SPLIT_NAME="test"
    # RANDOM CLASS
    #         Precision: 0.15313012443329796
    #         Recall: 0.5035685963521015
    #         F1: 0.2348459632382854
    #         F0.5: 0.17788908685469684
    #         MCC: 0.0031137110246290704
    #
    # MAJORITY CLASS
    #         Precision: 0.15201195843479517
    #         Recall: 1.0
    #         F1: 0.2639069105520907
    #         F0.5: 0.18305818956757036
    #         MCC: 0.0
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.5317119694802098
#         Recall: 0.35368754956383824
#         F1: 0.42480236212972666
#         F0.5: 0.48308132229972706
#         MCC: 0.35470364907091745

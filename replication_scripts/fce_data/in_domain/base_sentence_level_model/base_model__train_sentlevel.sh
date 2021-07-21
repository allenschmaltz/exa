################################################################################
#### uniCNN+BERT: Train the initial model using sentence-level labels
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

# These forward_type designations are no longer used in the code, but they are
# used to organize the directories of some of the original models from 2019, so
# I have left them here:
FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
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
--number_of_filter_maps="${FILTER_NUMS}" >"${OUTPUT_LOG_DIR}"/train.fce_only_v6.epoch.detection.v2.refactor.log.txt


# At the moment, sentence-level tuning is not implemented in the main code,
# but we can just save all of the epochs and then run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20


    # Max F1: 0.8789986091794159; epoch 15 offset 0.26530612244897955
    # Max F0.5: 0.883480825958702; epoch 9 offset 0.0
    # Max MCC: 0.6643063707183218; epoch 15 offset 0.061224489795918366

# Note that this is tuning at the *sentence-level*; token-level labels have
# not yet been considered at this point.
# In the paper, we used the F1 epoch as the epoch for zero-shot labeling and
# then subsequent min-max fine-tuning. In these settings, we do not have the
# liberty of retrospectively choosing another epoch based on the token-
# level metrics. In the supervised setting, we used the F0.5 epoch as the
# starting epoch based on the subsequent token-level metrics, but the
# difference was not particularly large.


################################################################################
#######  TEST (sentence level) uniCNN+BERT
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the FCE test set.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
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


#   (Accuracy from random prediction (only for debugging purposes): 0.4900735294117647)
#   (Accuracy from all 1's prediction (only for debugging purposes): 0.6790441176470589)
#   Ground-truth Stats: Number of instances with class 1: 1847 out of 2720
# test acc: 0.8088235294117647

# To get the particular value for the decision boundary determined above,
# run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/fce.sentence_level_score_vals.test.epoch15.txt \
--start_offset 0.26530612244897955 \
--end_offset 0.26530612244897955 \
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
      #
      #
      # Offset: 0.26530612244897955
      #         Predicted error count: 1813
      #         Precision: 0.8709321566464424
      #         Recall: 0.8548998375744451
      #         F1: 0.8628415300546448
      #         F0.5: 0.867677766787559
      #         MCC: 0.5811275937464756


################################################################################
#### uniCNN+BERT;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting, producing results
#### on the test split desired.
#### Note that by default for reference/analysis, we show the results for
#### varying the decision boundary, but the real-world scenario is the result
#### corresponding to a decision boundary of 0. We consider tuning this value
#### with a small number of token-labeled sentences in a subsequent section
#### below.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"

#### Choose the dev or test split, as desired -- typically for the zero-shot setting, we're only interested in test
SPLIT_NAME="test"
#SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
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


#     EPOCH="15"
#           # dev
#           DETECTION OFFSET: 0.0
#           GENERATED:
#                   Precision: 0.4724953494552219
#                   Recall: 0.38318965517241377
#                   F1: 0.42318219683446395
#                   F0.5: 0.4514523664432257
#                   MCC: 0.34689114124264464
#
#           # test
#           DETECTION OFFSET: 0.0
#           GENERATED:
#                   Precision: 0.47672023073753605
#                   Recall: 0.3670103092783505
#                   F1: 0.4147325029124473
#                   F0.5: 0.44982698961937717
#                   MCC: 0.3292551217434918


################################################################################
#### Tuning on sample from the dev set: uniCNN+BERT;
#### This was the +1k result in the paper.
#### The point here is that with a small amount of labeled data,
#### we can tune the decision boundary to, for example, increase precision.
#### This provides a point of comparison with some of the alternatives
#### considered (namely, matching against the support set as with
#### the exemplar auditing rules).
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FORWARD_TYPE=0

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"

SAMPLE_SIZE=1000

#SPLIT_NAME="test"
SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/samples/fce-public.dev.original.binaryevalformat.txt.sample${SAMPLE_SIZE}.txt \
--test_seq_labels_file ${DATA_DIR}/samples/fce-public.dev.original.binaryevalformat.labels.txt.sample${SAMPLE_SIZE}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/eval_binary_train.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt.sample${SAMPLE_SIZE}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval_binary_train.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.sample${SAMPLE_SIZE}.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval_binary_train.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt.sample${SAMPLE_SIZE}.txt \
--detection_offset 0 \
--fce_eval \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}"

# From this small subset of the dev set, we can determine the offset cutoff
# to consider:
# DETECTION OFFSET: 1.6000000000000014
# GENERATED:
#         Precision: 0.6544566544566545
#         Recall: 0.25621414913957935
#         F1: 0.36825833047062867
#         F0.5: 0.4992548435171386
#         MCC: 0.3587032633916275

# Then we can re-run the code from the previous section and read off the
# result associated with the applicable offset:

#     EPOCH="15"
#           # dev
#           DETECTION OFFSET: 1.6000000000000014
#           GENERATED:
#                   Precision: 0.6473509933774835
#                   Recall: 0.25280172413793106
#                   F1: 0.36360818350898955
#                   F0.5: 0.4933546433378197
#                   MCC: 0.35407254400811583
#
#           # test
#           DETECTION OFFSET: 1.6000000000000014
#           GENERATED:
#                   Precision: 0.6389372822299652
#                   Recall: 0.23267248215701825
#                   F1: 0.3411231252179979
#                   F0.5: 0.47356188262637994
#                   MCC: 0.32830447652427225

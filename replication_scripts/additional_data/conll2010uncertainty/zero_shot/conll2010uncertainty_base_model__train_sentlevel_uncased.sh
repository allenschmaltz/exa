################################################################################
#### uniCNN+BERT_{base_uncased}: Train the initial model using sentence-level labels
#### on the CONLL2010 data. Note that here we are using BERT_{base_uncased} for
#### reference to existing work.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}


OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir -p "${MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/data/glove.6B.300d.txt.word2vec_format.txt \
--word_embeddings_file_in_plaintext \
--training_file ${DATA_DIR}/conll2010uncertainty.train.txt \
--dev_file ${DATA_DIR}/conll2010uncertainty.dev.txt \
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
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case >"${OUTPUT_LOG_DIR}"/train.conll2010_document_level.log.txt


# At the moment, sentence-level tuning is not implemented in the main code,
# but we can just save all of the epochs and then run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

    # Percent of gold sentences with errors: 0.24387755102040817 (478 out of 1960)
    # ------------------------------------------------------------
    # RANDOM
    #         Predicted error count: 1005
    #         Precision: 0.23582089552238805
    #         Recall: 0.49581589958158995
    #         F1: 0.3196223870532704
    #         F0.5: 0.26345042240996
    #         MCC: -0.019246606179785356
    # ------------------------------------------------------------
    # MAJORITY CLASS
    #         Predicted error count: 1960
    #         Precision: 0.24387755102040817
    #         Recall: 1.0
    #         F1: 0.3921246923707958
    #         F0.5: 0.2873286847799952
    #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
    #         MCC: 0.0
    #
    # Max F1: 0.9189765458422176; epoch 10 offset 0.7551020408163265
    # Max F0.5: 0.9411500449236299; epoch 10 offset 0.4693877551020408
    # Max MCC: 0.8937869911096198; epoch 10 offset 0.7551020408163265


# Note that this is tuning at the *sentence-level*; token-level labels have
# not yet been considered at this point.
# In the paper, we used the F1 epoch as the epoch for zero-shot labeling and
# then subsequent min-max fine-tuning. In these settings, we do not have the
# liberty of retrospectively choosing another epoch based on the token-
# level metrics.

################################################################################
#######  TEST (sentence level) uniCNN+BERT_{base_uncased}
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the CONLL2010 test set.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="10"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.test.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${MODEL_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/conll2010.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case >"${OUTPUT_LOG_DIR}"/conll2010.test.sentence_level_score_vals.log.txt



# To get the particular value for the decision boundary determined above,
# run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/conll2010.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.7551020408163265 \
--end_offset 0.7551020408163265 \
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
      # ------------------------------------------------------------
      # MAJORITY CLASS
      #         Predicted error count: 1937
      #         Precision: 0.21631388745482705
      #         Recall: 1.0
      #         F1: 0.35568760611205436
      #         F0.5: 0.25652014203501894
      #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
      #         MCC: 0.0
      # ------------------------------------------------------------
      # Offset: 0.7551020408163265
      #         Predicted error count: 403
      #         Precision: 0.8982630272952854
      #         Recall: 0.863961813842482
      #         F1: 0.8807785888077858
      #         F0.5: 0.8911866075824717
      #         MCC: 0.8489430409854243




################################################################################
#### uniCNN+BERT_{base_uncased};
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting, producing results
#### on the test split.
#### Note that by default for reference/analysis, we show the results for
#### varying the decision boundary, but the real-world scenario is the result
#### corresponding to a decision boundary of 0.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="10"

#### Choose the dev or test split, as desired -- typically for the zero-shot setting, we're only interested in test
SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


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
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case

#SPLIT_NAME="dev"
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.43828125
#         Recall: 0.8577981651376146
#         F1: 0.5801447776628749
#         F0.5: 0.4857984066505023
#         MCC: 0.6063526333577466


#SPLIT_NAME="test"
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.42735042735042733
#         Recall: 0.8460236886632826
#         F1: 0.5678591709256103
#         F0.5: 0.4742933029785619
#         MCC: 0.5948359520187068


################################################################################
#### uniCNN+BERT_{base_uncased}+mm: uniCNN+BERT_{base_uncased} fine-tuned with the min-max loss
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

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune"
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
--mode "min_max_fine_tune" \
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
--score_vals_file "${OUTPUT_DIR}"/min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/min_max_fine_tune.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
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
--do_lower_case >"${OUTPUT_LOG_DIR}"/conll2010.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.log.txt



# At the moment, sentence-level tuning is not implemented in the refactored main code,
# but we can just save all of the epochs and then run the following:
# Note that --input_score_vals_file is the file prefix from above
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

      # Percent of gold sentences with errors: 0.24387755102040817 (478 out of 1960)
      # ------------------------------------------------------------
      #
      # RANDOM
      #         Predicted error count: 1005
      #         Precision: 0.23582089552238805
      #         Recall: 0.49581589958158995
      #         F1: 0.3196223870532704
      #         F0.5: 0.26345042240996
      #         MCC: -0.019246606179785356
      # ------------------------------------------------------------
      # MAJORITY CLASS
      #         Predicted error count: 1960
      #         Precision: 0.24387755102040817
      #         Recall: 1.0
      #         F1: 0.3921246923707958
      #         F0.5: 0.2873286847799952
      #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
      #         MCC: 0.0
      #
      #
      # Max F1: 0.7378129117259553; epoch 19 offset 1.0
      # Max F0.5: 0.8739076154806492; epoch 19 offset 1.0
      # Max MCC: 0.7169504088351373; epoch 19 offset 1.0



################################################################################
#######  TEST (sentence level) uniCNN+BERT_{base_uncased}+mm
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the CONLL2010 test set. Note that with the min-max loss,
####### using the top linear layer for the sentence-level prediction may not
####### be optimal, but this is for reference. (With sentence-level predictions,
####### we have dev labels, so we can choose whether to use token-level
####### projections, the top layer, etc. based on held-out dev.) We illustrate
####### this in the next section.
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

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="19"  # best in regard to sentence F1 from the fully-connected layer from above


BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.test.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/mm.conll2010.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--input_is_untokenized \
--data_formatter "fce" \
--do_lower_case >"${OUTPUT_LOG_DIR}"/mm.conll2010.test.sentence_level_score_vals.log.txt


# (Accuracy from random prediction (only for debugging purposes): 0.49561177077955604)
# (Accuracy from all 1's prediction (only for debugging purposes): 0.21631388745482705)
# Ground-truth Stats: Number of instances with class 1: 419 out of 1937
# test acc: 0.8699019101703666

# get F1:
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/mm.conll2010.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 1.0 \
--end_offset 1.0 \
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
    # ------------------------------------------------------------
    # MAJORITY CLASS
    #         Predicted error count: 1937
    #         Precision: 0.21631388745482705
    #         Recall: 1.0
    #         F1: 0.35568760611205436
    #         F0.5: 0.25652014203501894
    #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
    #         MCC: 0.0
    # ------------------------------------------------------------
    # Offset: 1.0
    #         Predicted error count: 243
    #         Precision: 0.9917695473251029
    #         Recall: 0.5751789976133651
    #         F1: 0.72809667673716
    #         F0.5: 0.8662832494608196
    #         MCC: 0.7133294057192907

# get F1 for decision boundary of 0---This is just for illustration to note that the fine-tuning
# has substantively shifted the parameters w.r.t. the document-level predictions. See the next section, as well.
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/mm.conll2010.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.0 \
--end_offset 0.0 \
--number_of_offsets 1

# Offset: 0.0
#         Predicted error count: 167
#         Precision: 1.0
#         Recall: 0.39856801909307876
#         F1: 0.5699658703071673
#         F0.5: 0.7681692732290709
#         MCC: 0.5846560884814858

#
################################################################################
#### uniCNN+BERT_{base_uncased}+mm; For reference, we get the sentence-level scores by
#### the token-level logits (compare with the section above). See Footnote 5
#### of the Online Appendix.
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
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="19"  # best in regard to sentence F1 from the fully-connected layer from above

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
--score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.test_by_contributions.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.test_by_contributions.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.test_by_contributions.txt \
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

# (Accuracy from the global fc (primarily for debugging purposes): 0.8699019101703666)
# test acc: 0.9468249870934434

#get the F1 score
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_for_minmax.py \
--input_score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.test_by_contributions.txt \
--start_offset 0.0 \
--end_offset 0.0 \
--number_of_offsets 1

# Offset: 0.0
#         Predicted error count: 402
#         Precision: 0.8930348258706468
#         Recall: 0.8568019093078759
#         F1: 0.8745432399512789
#         F0.5: 0.8855451406018746
#         MCC: 0.8411146054123988
# Accuracy: 0.9468249870934434



################################################################################
#### uniCNN+BERT_{base_uncased}+mm;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting.
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
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="19"  # best in regard to sentence F1 from the fully-connected layer from above

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
--do_lower_case

#SPLIT_NAME="test"
# RANDOM CLASS
#         Precision: 0.012981556946603369
#         Recall: 0.5228426395939086
#         F1: 0.02533409854882348
#         F0.5: 0.016126843627026294
#         MCC: 0.005140544799310995
#
# MAJORITY CLASS
#         Precision: 0.012412315705464779
#         Recall: 1.0
#         F1: 0.024520277979462712
#         F0.5: 0.015467398074819285
#         MCC: 0.0

# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.8669438669438669
#         Recall: 0.7055837563451777
#         F1: 0.7779850746268657
#         F0.5: 0.8290258449304175
#         MCC: 0.7796934029868926

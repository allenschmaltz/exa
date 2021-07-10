################################################################################
#### uniCNN+BERT_base: Train the initial model using sentence-level labels
#### on the BEA 2019 data. Note that here we are using BERT_base for
#### reference to existing work. We're using the splits from the original
#### work, wherein the dev split is carved off from train, and the "test"
#### set is the original dev set.
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

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
--training_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.train.txt \
--dev_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.dev.txt \
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
--number_of_filter_maps="${FILTER_NUMS}" >"${OUTPUT_LOG_DIR}"/train.bea2019_document_level.log.txt


# At the moment, sentence-level tuning is not implemented in the main code,
# but we can just save all of the epochs and then run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/train.score_vals.epoch.pt.dev.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

    # Percent of gold sentences with errors: 0.6779411764705883 (2305 out of 3400)
    # ------------------------------------------------------------
    # RANDOM
    #         Predicted error count: 1748
    #         Precision: 0.6893592677345538
    #         Recall: 0.5227765726681128
    #         F1: 0.5946212681963978
    #         F0.5: 0.6480585134989783
    #         MCC: 0.02513596686620451
    # ------------------------------------------------------------
    # MAJORITY CLASS
    #         Predicted error count: 3400
    #         Precision: 0.6779411764705883
    #         Recall: 1.0
    #         F1: 0.8080631025416302
    #         F0.5: 0.7246149009745363
    #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
    #         MCC: 0.0

    # Max F1: 0.8649885583524028; epoch 13 offset 0.26530612244897955
    # Max F0.5: 0.8661887694145758; epoch 6 offset 0.0
    # Max MCC: 0.5627351742337234; epoch 6 offset 0.0



# Note that this is tuning at the *sentence-level*; token-level labels have
# not yet been considered at this point.
# In the paper, we used the F1 epoch as the epoch for zero-shot labeling and
# then subsequent min-max fine-tuning. In these settings, we do not have the
# liberty of retrospectively choosing another epoch based on the token-
# level metrics.

################################################################################
#######  TEST (sentence level) uniCNN+BERT_base
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the test set.
################################################################################


SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${MODEL_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="13"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt \
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



# (Accuracy from random prediction (only for debugging purposes): 0.5139142335766423)
# (Accuracy from all 1's prediction (only for debugging purposes): 0.6514598540145985)
# Ground-truth Stats: Number of instances with class 1: 2856 out of 4384
# test acc: 0.7887773722627737

# To get the particular value for the decision boundary determined above,
# run the following:

python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/fce.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.26530612244897955 \
--end_offset 0.26530612244897955 \
--number_of_offsets 1

    # Percent of gold sentences with errors: 0.6514598540145985 (2856 out of 4384)
    # ------------------------------------------------------------
    # RANDOM
    #         Predicted error count: 2241
    #         Precision: 0.6497099509147702
    #         Recall: 0.5098039215686274
    #         F1: 0.5713164606631352
    #         F0.5: 0.6159052453468697
    #         MCC: -0.003755377587455455
    # ------------------------------------------------------------
    # MAJORITY CLASS
    #         Predicted error count: 4384
    #         Precision: 0.6514598540145985
    #         Recall: 1.0
    #         F1: 0.7889502762430939
    #         F0.5: 0.7002746174970577
    #         Warning: denominator in mcc calculation is 0; setting denominator to 1.0
    #         MCC: 0.0
    # ------------------------------------------------------------
    # Offset: 0.26530612244897955
    #         Predicted error count: 3365
    #         Precision: 0.7809806835066865
    #         Recall: 0.9201680672268907
    #         F1: 0.8448802443337085
    #         F0.5: 0.8053444471684237
    #         MCC: 0.4939407151493426



################################################################################
#### uniCNN+BERT_base;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting, producing results
#### on the test split.
#### Note that by default for reference/analysis, we show the results for
#### varying the decision boundary, but the real-world scenario is the result
#### corresponding to a decision boundary of 0.
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa

GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="13"

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
--test_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt \
--test_seq_labels_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.labels.test.txt \
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

# RANDOM CLASS
#         Precision: 0.10082993259663442
#         Recall: 0.5001705902422381
#         F1: 0.1678273644845548
#         F0.5: 0.1199901781573132
#         MCC: -0.0008493663352543561
#
# MAJORITY CLASS
#         Precision: 0.10108523210631595
#         Recall: 1.0
#         F1: 0.1836101859489032
#         F0.5: 0.12324205226819757
#         MCC: 0.0


# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.3726197183098592
#         Recall: 0.3760946207210281
#         F1: 0.37434910572786967
#         F0.5: 0.37330955229946045
#         MCC: 0.3036336937893875


################################################################################
#### uniCNN+BERT_base+mm: uniCNN+BERT_base fine-tuned with the min-max loss
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="13"  # this is for the original model; best in regard to sentence F1
#SPLIT_NAME="test"
FINE_TUNE_SPLIT_NAME="train"
SPLIT_NAME="dev"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'
#CUDA_VISIBLE_DEVICES=${GPU_IDS}

DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "min_max_fine_tune" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--training_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.train.txt \
--dev_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.dev.txt \
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
--training_seq_labels_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.labels.train.txt \
--dev_seq_labels_file ${DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.labels.dev.txt \
--forward_type 1 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} >"${OUTPUT_LOG_DIR}"/bea2019.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.log.txt


#Sentence-level max dev accuracy: 0.7952941176470588 at epoch 5

# At the moment, sentence-level tuning is not implemented in the refactored main code,
# but we can just save all of the epochs and then run the following:
# Note that --input_score_vals_file is the file prefix from above
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

    # Max F1: 0.8569016462642466; epoch 5 offset 0.8775510204081632
    # Max F0.5: 0.8755033191859831; epoch 3 offset 0.32653061224489793
    # Max MCC: 0.5613133827433359; epoch 4 offset 0.3877551020408163


################################################################################
#######  TEST (sentence level) uniCNN+BERT_base+mm
####### Here, for reference, we produce the *sentence-level* accuracy
####### on the test BEA2019 test set. Note that with the min-max loss,
####### using the top linear layer for the sentence-level prediction may not
####### be optimal, but this is for reference. (With sentence-level predictions,
####### we have dev labels, so we can choose whether to use token-level
####### projections, the top layer, etc. based on held-out dev.) We illustrate
####### this in the next section.
################################################################################


SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

#mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="5"  # best in regard to sentence F1 from the fully-connected layer from above


BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file not_used.txt \
--test_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/mm.bea2019.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" >"${OUTPUT_LOG_DIR}"/mm.bea2019.test.sentence_level_score_vals.log.txt


# (Accuracy from random prediction (only for debugging purposes): 0.4958941605839416)
# (Accuracy from all 1's prediction (only for debugging purposes): 0.6514598540145985)
# Ground-truth Stats: Number of instances with class 1: 2856 out of 4384
# test acc: 0.7817062043795621

# get F1:
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune.py \
--input_score_vals_file "${OUTPUT_DIR}"/mm.bea2019.sentence_level_score_vals.test.epoch${EPOCH}.txt \
--start_offset 0.8775510204081632 \
--end_offset 0.8775510204081632 \
--number_of_offsets 1

# Offset: 0.8775510204081632
#         Predicted error count: 3251
#         Precision: 0.7908335896647185
#         Recall: 0.9002100840336135
#         F1: 0.8419846078270837
#         F0.5: 0.810529634300126
#         MCC: 0.4954550247666714


################################################################################
#### uniCNN+BERT_base+mm; For reference, we get the sentence-level scores by
#### the token-level logits (compare with the section above)
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

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

EPOCH="5"  # best in regard to sentence F1 from the fully-connected layer from above

SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'
#CUDA_VISIBLE_DEVICES=${GPU_IDS}

DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "test_by_contributions" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt \
--test_seq_labels_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.labels.test.txt \
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
--dropout_probability ${DROPOUT_PROB}

SPLIT_NAME="test"


# sw_non-static_5.pt loaded successfully!
#         (Accuracy from the global fc (primarily for debugging purposes): 0.7817062043795621)
# test acc: 0.7634580291970803
# get the F1 score
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_for_minmax.py \
--input_score_vals_file "${OUTPUT_DIR}"/eval.min_max_fine_tune.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.test_by_contributions.txt \
--start_offset 0.0 \
--end_offset 0.0 \
--number_of_offsets 1

# Offset: 0.0
#         Predicted error count: 2523
#         Precision: 0.8604835513277844
#         Recall: 0.7601540616246498
#         F1: 0.8072132366610895
#         F0.5: 0.8383534136546186
#         MCC: 0.5107527703908336
# Accuracy: 0.7634580291970803

# Note the higher precision and lower recall (vis-a-vis, for example, the F1-document-level-dev-tuned result above),
# and a higher overall MCC. (Note that with an offset/decision-boundary of 0, this is the same as using the presence of at least
# 1 class 1 token-level prediction as determining a class 1 *document-level* label.)
# Note that we can also tune the decision boundary here, as above, but this is
# just for reference/illustration, and for consitency with the FCE results, we just use the linear layer's
# prediction for these grammar sets. In contrast, the CONLL2010 dataset provides an example (albeit
# somewhat artificial) as to when this might actually be preferable in practice for a particular
# label/data distribution. See Footnote 5 in the Online Appendix, and
# I provide a further discussion of these points, and variations on this
# theme, in the multi-label paper. For multi-label/multi-class cases, adding a global constraint may well be
# the default approach (unlike in these binary classification cases), but the main take-away is that we now have
# a few options for relating the token- and document- level predictions to each other, and the choice
# of which is preferable (and how direct that connection should be) will depend on the particular task and the metric of practical interest (e.g.,
# whether the end-application is more concerned with high[er] precision or high[er] recall for the
# document-level predictions, and whether it makes semantic sense for 1 token to fully describe the document-level prediction).
# (Since that decision can be made
# on the basis of the document-level labels, it is straightforward to do so by examining results on the dev set, as with standard
# classification settings.)


################################################################################
#### uniCNN+BERT_base+mm;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting.
################################################################################

SERVER_DATA_DIR="UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat"
SERVER_SCRATCH_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH
SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/exa


GPU_IDS=0

DATA_DIR=${SERVER_DATA_DIR}
VOCAB_SIZE=7500
FILTER_NUMS=1000
EXPERIMENT_LABEL=bert_unicnn_bea2019_v${VOCAB_SIZE}k_bertbasecased_top4layers_fn${FILTER_NUMS}_document_level

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

EPOCH="5"  # best in regard to sentence F1 from the fully-connected layer from above

SPLIT_NAME="test"
#SPLIT_NAME="dev"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'
#CUDA_VISIBLE_DEVICES=${GPU_IDS}

DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "zero" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt \
--test_seq_labels_file ${DATA_DIR}/ABCN.dev.gold.bea19.m2.labels.test.txt \
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
--dropout_probability ${DROPOUT_PROB}



# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.4518395267147347
#         Recall: 0.27794836802001593
#         F1: 0.3441768764962681
#         F0.5: 0.40159058792599156
#         MCC: 0.29962255901030865

# Note that the file "${OUTPUT_DIR}"/eval.min_max_fine_tune.viz.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html, which will be around 26mb, visualizes the detections ('PRED'), juxtaposed with the ground-truth labels ('GOLD'). Red highlights indicate predictions of the model and true detections for the ground-truth lines. This is easiest to view from a browser. Note that text-based (non-html) vizualization files, which are more amenable to reading and searching from a terminal, can be created by going through the K-NN data structures. (See the replication scripts for the CoNLL data, or those for the experiments in the main text for examples.)

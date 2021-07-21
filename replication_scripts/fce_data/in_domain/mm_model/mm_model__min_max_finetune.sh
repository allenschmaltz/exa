################################################################################
#### uniCNN+BERT+mm: uniCNN+BERT fine-tuned with the min-max loss
#### To fine-tune the model with the min-max loss, we initialize the parameters
#### with the uniCNN+BERT model (based on sentence-level F1).
#### In this case, the model still only has
#### access to the sentence-level labels. The fine-tuning method only uses the
#### sentence-level labels, but a label file containing an int for every token
#### is expected to be supplied to --training_seq_labels_file and
#### --dev_seq_labels_file. These can be random; they just have to have
#### the same number of tokens as the input as they are used to create padding
#### masks.
#### In any case, for a reference run on a token-labeled dataset, you can just
#### supply the original token-level files, as the token-level labels
#### will not be accessed by the train/eval.
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

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${FINE_TUNE_MODE_DIR}"
mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="15"  # this is for the original model; best in regard to sentence F1
#SPLIT_NAME="test"
FINE_TUNE_SPLIT_NAME="train"
SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


DROPOUT_PROB=0.50

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "min_max_fine_tune" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--training_file ${DATA_DIR}/fce-public.${FINE_TUNE_SPLIT_NAME}.original.binaryevalformat.txt \
--dev_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--save_model \
--epoch 20 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${FINE_TUNE_MODE_DIR}" \
--saved_model_file ${MODEL_DIR}/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/art_label_fine_tune_on_train.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/art_label_fine_tune_on_train.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/art_label_fine_tune_on_train.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--training_seq_labels_file ${DATA_DIR}/artificial_sent_projection/fce-public.train.original.binaryevalformat.labels.artificial_sent_projection.txt \
--dev_seq_labels_file ${DATA_DIR}/fce-public.${SPLIT_NAME}.original.binaryevalformat.labels.txt \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB} >"${OUTPUT_LOG_DIR}"/art_label_fine_tune_on_train.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.log.txt


# The question arises then as to which epoch to choose, since we don't have
# access to any token-level labels. There are a few options,
# and it will depend on the dataset/etc. One is to use the loss; one is to
# use the sentence-level accuracy from the fully-connected layer (which is
# what we did for the paper); another is to use the sentence-level
# accuracy calculated from the local labels (i.e., a positive prediction
# if at least one token is a positive prediction). If you use the
# local-to-global scores, it may be helpful to
# make the determination by analyzing the decision boundary. As an aside, note that the
# 'local-positive makes a global-positive' is an assumpition that doesn't make
# sense for all tasks. For example, if this method is used for class-conditional
# feature detection, then for some tasks, it may make sense to allow a small number
# of local activations/features of the other class. See my multi-label paper
# for a further discussion and additional possible priors. (That we can readily
# manipulate the token activations and still get similar global predictions is
# at the crux as to why we need to approach neural network interpretability/
# explainability with the 'actionable'/introspective approaches I am proposing.)
# Here, we choose the epoch
# based on the fully-connected layer, as with the original sentence-level
# trained model.

# At the moment, sentence-level tuning is not implemented in the refactored main code,
# but we can just save all of the epochs and then run the following:
# (Similarly, If you use the local-to-global scores, it may be helpful to
# make the determination by analyzing the decision boundary.)
# Note that --input_score_vals_file is the file prefix from above
python ${REPO_DIR}/code/eval/identification_of_binary_cnn_model_eval_fscore_and_mcc_tune_across_epochs.py \
--input_score_vals_file "${OUTPUT_DIR}"/art_label_fine_tune_on_train.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.cpu.refactor.txt \
--start_epoch 2 \
--end_epoch 20

# Here are the results from the model trained in 2019:
# Max F1: 0.8602073650339649; epoch 14 offset 0.9795918367346939
# Max F0.5: 0.8759309410968179; epoch 10 offset 0.9795918367346939
# Max MCC: 0.623261483075666; epoch 14 offset 0.9795918367346939

# I re-ran the fine-tuning with the newest version of Pytorch 1.8 and got the
# following:

# Max F1: 0.8700906344410876; epoch 12 offset 1.0
# Max F0.5: 0.8794433266264253; epoch 11 offset 0.9591836734693877
# Max MCC: 0.6186467019660382; epoch 10 offset 1.0

# The resulting token-level score evaluated below was effectively the same after re-training.

# One last point: I've subsequently added another option:
# --do_not_allow_padding_to_participate_in_min_max_loss
# which eliminates all edge cases in which a padding token participates in the
# min-max. This would seem to make sense if your problem calls for an even stronger
# sparsity bias, or if the base sentence-level classifier produces positive
# predictions for every token in some sentences,
# because in the original case, the padding tokens in effect serve as
# floors/ceilings in the min-max (akin to applying a relu). However, in practice,
# at least for 20 epochs (but I note that the loss was still decreasing)
# it turns out to result in a more muted rise in precision on the FCE set,
# but it may be worth considering for other datasets. (I used a variation on this
# theme for the later multi-label variant, which I've subsequently added back here
# for reference.)


################################################################################
#### uniCNN+BERT+mm;
#### Next, we produce scores at the token-level using the proposed
#### decomposition. This is the 'pure' zero-shot setting.
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

# this is the directory to the original sentence-level trained model
MODEL_DIR=${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/fce_cnn_interp/${EXPERIMENT_LABEL}

#FORWARD_TYPE=2
FORWARD_TYPE=1

FINE_TUNE_MODE_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/models/grammar/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/debug_cnn_finetune/${EXPERIMENT_LABEL}_ftype${FORWARD_TYPE}_zerofinetune_epoch15/out"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"

EPOCH="14"  # best in regard to sentence F1 from the fully-connected layer from above

SPLIT_NAME="test"
#SPLIT_NAME="dev"

#BERT_MODEL="bert-base-cased"
BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'


DROPOUT_PROB=0.50

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
--saved_model_file "${FINE_TUNE_MODE_DIR}"/aesw_non-static_${EPOCH}.pt \
--score_vals_file "${OUTPUT_DIR}"/eval_label_fine_tune_on_add_loss.score_vals.epoch${EPOCH}.pt.${SPLIT_NAME}.txt.gpu.refactor.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/data/mltagger/pytorch_pretrained_bert" \
--bert_model ${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--color_gradients_file ${REPO_DIR}/code/support_files/pink_to_blue_64.txt \
--visualization_out_file "${OUTPUT_DIR}"/eval_label_fine_tune_on_add_loss.fce_only_v5.epoch${EPOCH}.v2.${SPLIT_NAME}.refactor.html \
--correction_target_comparison_file ${MODEL_DIR}/not_used.txt \
--output_generated_detection_file "${OUTPUT_DIR}"/eval_label_fine_tune_on_add_loss.fce_only_v6.epoch${EPOCH}.${SPLIT_NAME}.detection.v2.refactor.txt \
--detection_offset 0 \
--fce_eval \
--forward_type ${FORWARD_TYPE} \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB}



# #SPLIT_NAME="test"
# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.548744019138756
#         Recall: 0.29103885804916735
#         F1: 0.3803502953673956
#         F0.5: 0.46618566129769834
#         MCC: 0.32722479531357185

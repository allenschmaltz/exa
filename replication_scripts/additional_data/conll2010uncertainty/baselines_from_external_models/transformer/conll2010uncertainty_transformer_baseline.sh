################################################################################
#### Dependencies for code from Bujel, Yannakoudakis, and Rei (2021): https://github.com/bujol12/bert-seq-interpretability
################################################################################

# I used the latest commit as of writing, eed39dceff3560ea3311bc2540671783414e3cce, at https://github.com/bujol12/bert-seq-interpretability.
# The config files are included in this repo. You will need to update UPDATE_WITH_YOUR_PATH, as applicable. Additionally, you need to modify
#
# bert-seq-interpretability-master/utils/arguments.py
#
# with the new path and filenames for conll2010 (an absoluate path in place of UPDATE_WITH_YOUR_PATH will work), as created in additional_data/conll2010uncertainty/conll2010uncertainty_data_init.sh (see below section on 'DATA' for this particular directory structure):
#
    # conll10=DataTrainingArguments(
    #     name="conll10",
    #     data_dir="UPDATE_WITH_YOUR_PATH/data/conll10",
    #     labels="UPDATE_WITH_YOUR_PATH/data/conll10/conll2010uncertainty.conllformat_monolabels.train.txt",
    #     file_name="conll2010uncertainty.conllformat_monolabels.{mode}.txt",
    #     file_name_token="conll2010uncertainty.conllformat.{mode}.txt",
    #     positive_label="U",
    # ),
#
# in place of the following (from the original file)
#
    # conll10=DataTrainingArguments(
    #     name="conll10",
    #     data_dir="data/conll10",
    #     labels="data/conll10/conll10_task2_rev2.cue.train.tsv_sentencelevel",
    #     file_name="conll10_task2_rev2.cue.{mode}.tsv_sentencelevel",
    #     file_name_token="conll10_task2_rev2.cue.{mode}.tsv",
    #     positive_label="C",
    # ),

# I ran the code with Pytorch '1.7.0' with Cuda 10.2.

# using CUDA 10.2
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
pip install numpy==1.18.5
pip install scikit-learn==0.22.2

# For training, I used Huggingface Transformers (Release: v3.5.1).

# install Huggingface Transformers (Release: v3.5.1) -- used for training
cd UPDATE_WITH_YOUR_PATH/transformers-d5b3e56de5376aa85ef46e7f0325139d9e299a41
pip install -e .

# For training, I modified the following line (Line 563) in the function <compute_seq_classification_metrics(p: EvalPrediction) -> Dict> in bert-seq-interpretability-master/utils/tsv_dataset.py:
#
# 563: preds_list = np.argmax(p.predictions, axis=1).astype(int)
# changing to
# 563: preds_list = np.argmax(p.predictions[0], axis=1).astype(int)
#
# in order to calcuate the argmax over the document-level predictions (at least using the dependencies I used), where p.predictions[0] are the document-level predictions and p.predictions[1] are the token-level predictions.

# For eval, I dropped down to Huggingface Transformers (Release: v2.11.0) since v3.5.1 lacks DefaultDataCollator.
cd UPDATE_WITH_YOUR_PATH/transformers_2.11.0/transformers-2.11.0
pip install -e .


################################################################################
#### DATA
################################################################################

# To exactly match the directory structure used above, you can copy the files created in additional_data/conll2010uncertainty/conll2010uncertainty_data_init.sh as follows:

# move to combined directory
DATA_DIR=UPDATE_WITH_YOUR_PATH/data/conll10

cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat_monolabels/conll2010uncertainty.conllformat_monolabels.train.txt ${DATA_DIR}/
cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat_monolabels/conll2010uncertainty.conllformat_monolabels.dev.txt ${DATA_DIR}/
cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat_monolabels/conll2010uncertainty.conllformat_monolabels.test.txt ${DATA_DIR}/
cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat/conll2010uncertainty.conllformat.train.txt ${DATA_DIR}/
cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat/conll2010uncertainty.conllformat.dev.txt ${DATA_DIR}/
cp UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat/conll2010uncertainty.conllformat.test.txt ${DATA_DIR}/

################################################################################
#### Train Bujel, Yannakoudakis, and Rei (2021) on this split/version of CONLL2010
#### Using transformers version v3.5.1
################################################################################

cd UPDATE_WITH_YOUR_PATH/bert-seq-interpretability-master

GPU_ID=0
OUTPUT_LOG_FILE=UPDATE_WITH_YOUR_PATH/output/train_conll10_conll10cue_for_new_uncertainty_splits_monolabels.log.txt

export TRANSFORMERS_CACHE=UPDATE_WITH_YOUR_PATH/transformers_cache/

# the config file is in ${REPO_DIR}/replication_scripts/additional_data/conll2010uncertainty/baselines_from_external_models/transformer/ and can be copied over to the bert-seq-interpretability-master configs directory, or another directory

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u seq_class_script.py configs/train_conll10_conll10cue_for_new_uncertainty_splits_monolabels.conf ${GPU_ID} > ${OUTPUT_LOG_FILE}

  # The final output will be along the lines of the following:
      # INFO:__main__:final_checkpoint name: UPDATE_WITH_YOUR_PATH/models/final_soft_attention/roberta-base/conll10/20210620T081813/checkpoint-18000
      # INFO:__main__:UPDATE_WITH_YOUR_PATH/models/final_soft_attention/roberta-base/conll10/20210620T081813/final_model/
      #
      # UPDATE_WITH_YOUR_PATH/models/final_soft_attention/roberta-base/conll10/20210620T081813/checkpoint-18000 = {'eval_loss': 0.3620540499687195, 'eval_precision': 0.8705357142857143, 'eval_recall': 0.9307875894988067, 'eval_f1': 0.8996539792387543, 'eval_accuracy': 0.9550851832731028}
      #
      #
      #
      # [dev]
      # {'eval_loss': 0.35332149267196655, 'eval_precision': 0.8994082840236687, 'eval_recall': 0.9539748953974896, 'eval_f1': 0.9258883248730965, 'eval_accuracy': 0.9627551020408164}
      # [test]
      # {'eval_loss': 0.3620540499687195, 'eval_precision': 0.8705357142857143, 'eval_recall': 0.9307875894988067, 'eval_f1': 0.8996539792387543, 'eval_accuracy': 0.9550851832731028}

################################################################################
#### Eval Bujel, Yannakoudakis, and Rei (2021) on this split/version of CONLL2010
#### Using transformers version v2.11.0
################################################################################

cd UPDATE_WITH_YOUR_PATH/bert-seq-interpretability-master

GPU_ID=0
OUTPUT_LOG_FILE=UPDATE_WITH_YOUR_PATH/output/model_to_token_preds_conll10_conll10cue_for_new_uncertainty_splits_monolabels.log.txt

#export TRANSFORMERS_CACHE=UPDATE_WITH_YOUR_PATH/transformers_cache/

# the config file is in ${REPO_DIR}/replication_scripts/additional_data/conll2010uncertainty/baselines_from_external_models/transformer/ and can be copied over to the bert-seq-interpretability-master configs directory, or another directory

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u model_to_token_preds.py configs/pred_conll10_conll10cue_for_new_uncertainty_splits_monolabels.conf ${GPU_ID} > ${OUTPUT_LOG_FILE}

# model_to_token_preds_conll10_conll10cue_for_new_uncertainty_splits_monolabels.log.txt should contain:
# 1937
# 1937

OUTPUT_LOG_FILE=UPDATE_WITH_YOUR_PATH/output/token_preds_evaluate_conll10_conll10cue_for_new_uncertainty_splits_monolabels.log.txt

# the config file is in ${REPO_DIR}/replication_scripts/additional_data/conll2010uncertainty/baselines_from_external_models/transformer/ and can be copied over to the bert-seq-interpretability-master configs directory, or another directory

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u token_preds_evaluate.py configs/test_conll10_conll10cue_for_new_uncertainty_splits_monolabels.conf > ${OUTPUT_LOG_FILE}

# {'precision': 0.27646454265159304, 'recall': 0.9103214890016921, 'f1': 0.42412297989751685, 'f0.5': 0.3211940298507463, 'MAP': 0.9076973735853663, 'corr': 0.5010659999386946}


# token_preds_evaluate_conll10_conll10cue_for_new_uncertainty_splits_monolabels.log.txt should contain similar to the following:
# 1937
# 1937
# {'predicted_cnt': 1946, 'correct_cnt': 538, 'total_cnt': 591}


# The output file UPDATE_WITH_YOUR_PATH/token_preds/soft_attention/final_soft_attention/roberta-base/conll10/{datetime}/token_scores.tsv will contain the predictions.
# For example, here is sentence 779 shown in Table G.2.

# The     0.0012399335391819477
# BCL6    0.0012460621073842049
# gene    0.0011765306117013097
# encodes 0.001365496078506112
# a       0.001321122283115983
# 95-kDa  0.0013877907767891884
# protein 0.0012567405356094241
# containing      0.0013641909463331103
# six     0.0013827880611643195
# C-terminal      0.0013501516077667475
# zinc-finger     0.0013752359664067626
# motifs  0.0013906353851780295
# and     0.0012706599663943052
# an      0.0013687890022993088
# N-terminal      0.0013013583375141025
# POZ     0.0012687962735071778
# domain, 0.9979873895645142
# suggesting      0.9981476068496704
# that    0.9981288313865662
# it      0.9034187197685242
# may     0.9982224106788635
# function        0.9973028898239136
# as      0.8384379744529724
# a       0.016408920288085938
# transcription   0.0018006560858339071
# factor. 0.0017252340912818909

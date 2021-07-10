################################################################################
#### create the binaryevalformat for the CONLL2010 uncertainty dataset
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/data/conll2010

DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty
OUTPUT_DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat
mkdir ${OUTPUT_DATA_DIR}

#BERT_MODEL="bert-large-cased"
BERT_MODEL="bert-base-cased"

python conll2010uncertainty_data_to_binaryevalformat.py \
--input_files "${DATA_DIR}/bio_bmc.xml,${DATA_DIR}/bio_fly.xml,${DATA_DIR}/bio_hbc.xml,${DATA_DIR}/factbank.xml" \
--eval_split_size_proportion 0.1 \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/main/models/bert_cache/started_2020_03_10/" \
--bert_model=${BERT_MODEL} \
--output_binaryevalformat_train_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.train.txt" \
--output_binaryevalformat_train_labels_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.labels.train.txt" \
--output_binaryevalformat_dev_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.dev.txt" \
--output_binaryevalformat_dev_labels_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.labels.dev.txt" \
--output_binaryevalformat_test_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.test.txt" \
--output_binaryevalformat_test_labels_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.labels.test.txt"

      # Total number of sentences: 2401
      # Total number of documents: 9
      # Class 0 count: 1925; Class 1 count: 476
      # Mean length: 23.775926697209496; std: 11.69783828218243; min: 2, max: 87
      # Number of class 0 tokens: 56373
      # Number of class 1 tokens: 713
      # Total tokens: 57086
      # Total number of sentences: 2991
      # Total number of documents: 10
      # Class 0 count: 2283; Class 1 count: 708
      # Mean length: 25.897024406552994; std: 13.298797007421634; min: 1, max: 124
      # Number of class 0 tokens: 76421
      # Number of class 1 tokens: 1037
      # Total tokens: 77458
      # Total number of sentences: 11577
      # Total number of documents: 1277
      # Class 0 count: 8621; Class 1 count: 2956
      # Mean length: 24.77204802625896; std: 10.356375216364134; min: 2, max: 150
      # Number of class 0 tokens: 282754
      # Number of class 1 tokens: 4032
      # Total tokens: 286786
      # Total number of sentences: 3129
      # Total number of documents: 208
      # Class 0 count: 2575; Class 1 count: 554
      # Mean length: 20.91946308724832; std: 9.957816182845894; min: 1, max: 69
      # Number of class 0 tokens: 64718
      # Number of class 1 tokens: 739
      # Total tokens: 65457
      # ------------------------
      # Total number of documents: 1504
      # Total sentences in test: 1940
      # Total sentences in dev: 1960
      # Total sentences in train: 16198
      # The following test sentence also appears in the train split, so it is removed from test:
      # 0 However, the mechanism by which IFNs mediate this inhibition has not been defined.
      #
      # The following test sentence also appears in the train split, so it is removed from test:
      # 0 WASHINGTON _
      #
      # The following test sentence also appears in the dev split, so it is removed from test:
      # 0 Copyright 1998 Academic Press.
      #
      # The following test sentence also appears in the train split, so it is removed from test:
      # 0 Copyright 1998 Academic Press.
      #
      # Total filtered sentences in test: 1937


################################################################################
#### convert the binaryevalformat data to conll tab format (as used by the
#### models in the previous works)
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/data

DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat
OUTPUT_DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat
mkdir ${OUTPUT_DATA_DIR}


for SPLIT_NAME in "train" "dev" "test";
do
python binaryevalformat_to_conllformat.py \
--input_binaryevalformat_file "${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt" \
--input_binaryevalformat_labels_file "${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt" \
--output_conllformat_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.conllformat.${SPLIT_NAME}.txt"
done


################################################################################
#### convert the binaryevalformat data to conll tab format -- here we
#### use monolabels just to double check that the code of the previous
#### models are not indirectly using the token labels
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/data

DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/binaryevalformat
OUTPUT_DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat_monolabels
mkdir ${OUTPUT_DATA_DIR}


for SPLIT_NAME in "train" "dev" "test";
do
python binaryevalformat_to_conllformat_monolabels.py \
--input_binaryevalformat_file "${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt" \
--input_binaryevalformat_labels_file "${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt" \
--output_conllformat_file "${OUTPUT_DATA_DIR}/conll2010uncertainty.conllformat_monolabels.${SPLIT_NAME}.txt"
done

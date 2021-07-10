################################################################################
#### create the binaryevalformat for the BEA2019 dataset
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}/code/data/bea2019

DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2
OUTPUT_DATA_DIR=UPDATE_WITH_YOUR_PATH/data/zero/wi+locness/m2/binaryevalformat
mkdir ${OUTPUT_DATA_DIR}

#BERT_MODEL="bert-large-cased"
BERT_MODEL="bert-base-cased"

python bea2019_data_to_binaryevalformat.py \
--input_train_file "${DATA_DIR}/ABC.train.gold.bea19.m2" \
--input_test_file "${DATA_DIR}/ABCN.dev.gold.bea19.m2" \
--output_binaryevalformat_train_file "${OUTPUT_DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.train.txt" \
--output_binaryevalformat_train_labels_file "${OUTPUT_DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.labels.train.txt" \
--output_binaryevalformat_dev_file "${OUTPUT_DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.dev.txt" \
--output_binaryevalformat_dev_labels_file "${OUTPUT_DATA_DIR}/ABC.train.gold.bea19.m2.90_10percent_split.labels.dev.txt" \
--output_binaryevalformat_test_file "${OUTPUT_DATA_DIR}/ABCN.dev.gold.bea19.m2.test.txt" \
--output_binaryevalformat_test_labels_file "${OUTPUT_DATA_DIR}/ABCN.dev.gold.bea19.m2.labels.test.txt" \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/main/models/bert_cache/started_2020_03_10/" \
--bert_model=${BERT_MODEL}

    # Ignoring  with label 1
    # WARNING: An unexpected character was seen by the BERT tokenizer, so dropping:
    # 	ORIGINAL: Hope this message finds you in good health 
    # 	CLEANED: Hope this message finds you in good health
    # 	CLEANED: 0 0 0 0 0 0 0 0
    # 	Manually fixing labels:
    # 	FIXED: 0 0 0 0 0 0 0 1
    # Class 0 count: 11263; Class 1 count: 23045
    # Mean length: 18.328028448175353; std: 12.049002160661667; min: 1, max: 220
    # Number of insertions: 15620
    # 	Number of insertions of UNK type: 63
    # 	Number of insertions at the end of the sentence: 359
    # Number of deletions: 6638
    # Number of replacements: 41425
    # Number of negations standardized: 78
    # Total tokens: 628798
    # Length of full training output: 34308
    # Splitting train to create held-out dev
    # Length of training split: 30908
    # Length of dev split: 3400
    # Class 0 count: 1528; Class 1 count: 2856
    # Mean length: 19.841697080291972; std: 12.631643819672552; min: 1, max: 157
    # Number of insertions: 1949
    # 	Number of insertions of UNK type: 8
    # 	Number of insertions at the end of the sentence: 29
    # Number of deletions: 764
    # Number of replacements: 4919
    # Number of negations standardized: 13
    # Total tokens: 86986

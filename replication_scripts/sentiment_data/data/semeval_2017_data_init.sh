################################################################################
#### Preprocessed SemEval-2017 Task 4a data
################################################################################

# Note that the preprocessed data is available via the link in README_replication_main_text.md.
# The following shows how the preprocessed data was created, but is
# unnecessary to re-run if the preprocessed data above is available for download.

################################################################################
#### Download the SemEval-2017 Task 4a data
################################################################################

# Note that we most recently checked these pre-processing scripts with a version
# of the data downloaded in the first half of 2020 from
# http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip.

################################################################################
#### Preprocess Twitter data (SemEval-2017 Task 4a English) as binaryevalformat
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"
DATA_DIR="UPDATE_WITH_YOUR_PATH/SemEval2017-task4-test/"


OUTPUT_DATA_DIR="${DATA_DIR}/binaryevalformat"
mkdir "${OUTPUT_DATA_DIR}"


INPUT_FILE="SemEval2017-task4-test.subtask-A.english.txt"

EXPECTED_NUM_FILEDS=3

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/semeval_2017/semeval_2017_task_4a_sentiment_data_to_binaryevalformat_v2.py \
--input_tsv_file "${DATA_DIR}/${INPUT_FILE}" \
--expected_number_of_fields ${EXPECTED_NUM_FILEDS} \
--output_binaryevalformat_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.txt" \
--output_monolabels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.monolabels.txt" \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/" \
--bert_model=${BERT_MODEL} \
--output_balanced_filtering_binaryevalformat_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.balanced.txt" \
--output_balanced_filtering_monolabels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.monolabels.balanced.txt"

# Negative count: 3972; Positive count: 2375
# Neutral count (ignored): 5937
# Mean length: 16.14652591775642; std: 5.446355702290578; min: 1, max: 32
# Reducing number of negative lines to 2375 in order to match the number of positive class instances.
#
# Length of output: 6347
# Length of balanced output: 4750

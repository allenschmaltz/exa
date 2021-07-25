################################################################################
#### Preprocessed data
################################################################################

# Note that the preprocessed data is available via the link in README_replication_main_text.md.
# The following shows how the preprocessed data was created, but is
# unnecessary to re-run if the preprocessed data above is available for download.

################################################################################
#### Download the counterfactually-augmented data
################################################################################

# Note that we most recently checked these pre-processing scripts with a version
# of the data downloaded on 2020-04-10 from
# https://github.com/acmi-lab/counterfactually-augmented-data.

################################################################################
#### Preprocess data as binaryevalformat
#### Note that we do not remove <br /><br /> and other artifacts
#### The reviews remain one document of sentences.
################################################################################



REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

## TODO: choose the directory prefix (${DIR_PREFIX}) AND indicate the
## expected number of fields (${EXPECTED_NUM_FILEDS}) AND then uncomment
## the applicable line of the for loop below. The expected output when
## processing all of these sets is provided in comments after the code block.

#DIR_PREFIX="combined"
#DIR_PREFIX="new"
#DIR_PREFIX="orig"
#DIR_PREFIX="combined/paired"
DIR_PREFIX="orig/eighty_percent"
INPUT_DATA_DIR="${DATA_DIR}/${DIR_PREFIX}"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/binaryevalformat"
mkdir "${OUTPUT_DATA_DIR}"

#INPUT_FILE="train.tsv"
#INPUT_FILE="dev.tsv"
#INPUT_FILE="test.tsv"
#INPUT_FILE="train_paired.tsv"
#EXPECTED_NUM_FILEDS=2
EXPECTED_NUM_FILEDS=3

#for INPUT_FILE in "train.tsv" "dev.tsv" "test.tsv";
for INPUT_FILE in "train.tsv" "test.tsv";
#for INPUT_FILE in "train_paired.tsv" "dev_paired.tsv" "test_paired.tsv";
do
  echo "INPUT_FILE=${INPUT_FILE}"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_to_binaryevalformat_v2.py \
--input_tsv_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--expected_number_of_fields ${EXPECTED_NUM_FILEDS} \
--output_binaryevalformat_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.txt" \
--output_monolabels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.monolabels.txt" \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/" \
--bert_model=${BERT_MODEL}

done

# DIR_PREFIX="combined"
# INPUT_FILE=train.tsv
# Negative count: 1707; Positive count: 1707
# Mean length: 162.46865846514353; std: 66.38955914671601; min: 14, max: 347
# Length of output: 3414
# INPUT_FILE=dev.tsv
# Negative count: 245; Positive count: 245
# Mean length: 165.02244897959184; std: 67.57974507959399; min: 32, max: 327
# Length of output: 490
# INPUT_FILE=test.tsv
# Negative count: 488; Positive count: 488
# Mean length: 162.3483606557377; std: 65.19901403962163; min: 23, max: 337
# Length of output: 976


# DIR_PREFIX="new"
# INPUT_FILE=train.tsv
# Negative count: 856; Positive count: 851
# Mean length: 161.93204452255418; std: 66.22302644028532; min: 14, max: 347
# Length of output: 1707
# INPUT_FILE=dev.tsv
# Negative count: 123; Positive count: 122
# Mean length: 164.58775510204083; std: 67.07380532120237; min: 33, max: 327
# Length of output: 245
# INPUT_FILE=test.tsv
# Negative count: 245; Positive count: 243
# Mean length: 162.00409836065575; std: 65.03904867120904; min: 23, max: 337
# Length of output: 488

# DIR_PREFIX="orig"
# INPUT_FILE=train.tsv
# Negative count: 851; Positive count: 856
# Mean length: 163.00527240773286; std: 66.55134850198843; min: 14, max: 330
# Length of output: 1707
# INPUT_FILE=dev.tsv
# Negative count: 122; Positive count: 123
# Mean length: 165.45714285714286; std: 68.0791496145839; min: 32, max: 325
# Length of output: 245
# INPUT_FILE=test.tsv
# Negative count: 243; Positive count: 245
# Mean length: 162.69262295081967; std: 65.35677454071636; min: 23, max: 328
# Length of output: 488



# DIR_PREFIX="combined/paired"
#
# INPUT_FILE=train_paired.tsv
# Negative count: 1707; Positive count: 1707
# Mean length: 162.44258933801993; std: 66.37063355203925; min: 14, max: 347
# Length of output: 3414
# INPUT_FILE=dev_paired.tsv
# Negative count: 245; Positive count: 245
# Mean length: 165.0265306122449; std: 67.58119311963613; min: 32, max: 327
# Length of output: 490
# INPUT_FILE=test_paired.tsv
# Negative count: 488; Positive count: 488
# Mean length: 162.3483606557377; std: 65.19901403962164; min: 23, max: 337
# Length of output: 976


# DIR_PREFIX="orig/eighty_percent"
# INPUT_FILE=train.tsv
# Negative count: 9631; Positive count: 9631
# Mean length: 163.28309625168725; std: 66.72113890445429; min: 10, max: 331
# Length of output: 19262
# INPUT_FILE=test.tsv
# Negative count: 10000; Positive count: 10000
# Mean length: 160.0625; std: 64.554141569306; min: 4, max: 315
# Length of output: 20000


################################################################################
#### Generate sequence labels from the paired data
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}


DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

DIR_PREFIX="combined/paired"
INPUT_DATA_DIR="${DATA_DIR}/${DIR_PREFIX}/binaryevalformat"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/sequence_labels/"
mkdir "${OUTPUT_DATA_DIR}"



for INPUT_FILE in "train_paired.tsv" "dev_paired.tsv" "test_paired.tsv";
do
  echo "INPUT_FILE=${INPUT_FILE}"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_create_sequence_labels_from_paired_data.py \
--input_paired_binaryevalformat_file "${INPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.txt" \
--output_sequence_labels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.sequence_labels.txt"

done


################################################################################
#### Separate and align the diff data from the combined data
#### Note that this primarily enables running eval on test with the sequence
#### labels restricted to the orig data (to compare orig+new vs. just orig in
#### detecting sentiment sequences)
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

DIR_PREFIX="combined/paired"
INPUT_DATA_DIR="${DATA_DIR}/${DIR_PREFIX}/binaryevalformat"
INPUT_SEQ_DATA_DIR="${INPUT_DATA_DIR}/sequence_labels/"
mkdir "${DATA_DIR}/orig/binaryevalformat/aligned_sequence_labels/"
mkdir "${DATA_DIR}/new/binaryevalformat/aligned_sequence_labels/"

INPUT_FILE="dev"
INPUT_FILE="test"
INPUT_FILE="train"
for INPUT_FILE in "train" "dev" "test";
do
  echo "INPUT_FILE=${INPUT_FILE}"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_separate_and_align_paired_data.py \
--input_split ${INPUT_FILE} \
--input_paired_binaryevalformat_file "${INPUT_DATA_DIR}/${INPUT_FILE}_paired.tsv.binaryevalformat.txt" \
--input_paired_sequence_labels_file "${INPUT_SEQ_DATA_DIR}/${INPUT_FILE}_paired.tsv.binaryevalformat.sequence_labels.txt" \
--input_orig_binaryevalformat_file "${DATA_DIR}/orig/binaryevalformat/${INPUT_FILE}.tsv.binaryevalformat.txt" \
--input_new_binaryevalformat_file "${DATA_DIR}/new/binaryevalformat/${INPUT_FILE}.tsv.binaryevalformat.txt" \
--output_orig_binaryevalformat_file "${DATA_DIR}/orig/binaryevalformat/aligned_sequence_labels/${INPUT_FILE}.aligned_binaryevalformat.txt" \
--output_new_binaryevalformat_file "${DATA_DIR}/new/binaryevalformat/aligned_sequence_labels/${INPUT_FILE}.aligned_binaryevalformat.txt" \
--output_orig_sequence_labels_file "${DATA_DIR}/orig/binaryevalformat/aligned_sequence_labels/${INPUT_FILE}.aligned_binaryevalformat.sequence_labels.txt" \
--output_new_sequence_labels_file "${DATA_DIR}/new/binaryevalformat/aligned_sequence_labels/${INPUT_FILE}.aligned_binaryevalformat.sequence_labels.txt"

done


################################################################################
#### Separate and align the diff data from the combined data to
#### create the data split by domain --- i.e., here, label 0 is the original
#### data and 1 is the new data (including for the sequence labels, where the
#### labels indicate diffs)
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

DIR_PREFIX="combined/paired"
INPUT_DATA_DIR="${DATA_DIR}/${DIR_PREFIX}/binaryevalformat"
INPUT_SEQ_DATA_DIR="${INPUT_DATA_DIR}/sequence_labels/"
mkdir "${DATA_DIR}/combined/paired/domain/"


for INPUT_FILE in "train" "dev" "test";
do
  echo "INPUT_FILE=${INPUT_FILE}"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_create_predict_diffs_data.py \
--input_paired_binaryevalformat_file "${INPUT_DATA_DIR}/${INPUT_FILE}_paired.tsv.binaryevalformat.txt" \
--input_orig_binaryevalformat_file "${DATA_DIR}/orig/binaryevalformat/${INPUT_FILE}.tsv.binaryevalformat.txt" \
--input_new_binaryevalformat_file "${DATA_DIR}/new/binaryevalformat/${INPUT_FILE}.tsv.binaryevalformat.txt" \
--output_paired_domain_binaryevalformat_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.paired.domain_binaryevalformat.txt" \
--output_paired_domain_sequence_labels_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.txt" \
--output_paired_domain_binaryevalformat_only_orig_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.only_orig.domain_binaryevalformat.txt" \
--output_paired_domain_sequence_labels_only_orig_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.only_orig.domain_binaryevalformat.sequence_labels.txt" \
--output_paired_domain_binaryevalformat_only_new_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.only_new.domain_binaryevalformat.txt" \
--output_paired_domain_sequence_labels_only_new_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.only_new.domain_binaryevalformat.sequence_labels.txt"

done


################################################################################
#### Create the 3.4k orig sample -- Here, we take the 1.7k orig and sample
#### and additional 1.7k disjoint reviews from the full orig
#### The size of 3414 comes from the paired combined train data.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

mkdir "${DATA_DIR}/orig/binaryevalformat/train3.4k/"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_create_orig_3k_sample.py \
--input_orig_binaryevalformat_file "${DATA_DIR}/orig/binaryevalformat/train.tsv.binaryevalformat.txt" \
--input_full_orig_binaryevalformat_file "${DATA_DIR}/orig/eighty_percent/binaryevalformat/train.tsv.binaryevalformat.txt" \
--final_sample_size 3414 \
--output_binaryevalformat_file "${DATA_DIR}/orig/binaryevalformat/train3.4k/train.orig_3.4k.binaryevalformat.txt"

#Number of filtered lines: 1711


################################################################################
#### Separate the domain data by sentiment
#### This allows us to analyze new/orig domain data conditional on the underlying assigned sentiment
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

OUTPUT_DIR="${DATA_DIR}/combined/paired/domain/separated_by_sentiment"
mkdir "${OUTPUT_DIR}"

INPUT_FILE="test"


python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_separate_paired_data_by_sentiment.py \
--input_paired_sentiment_binaryevalformat_file "${DATA_DIR}/combined/paired/binaryevalformat/${INPUT_FILE}_paired.tsv.binaryevalformat.txt" \
--input_paired_domain_binaryevalformat_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.paired.domain_binaryevalformat.txt" \
--input_paired_domain_sequence_labels_file "${DATA_DIR}/combined/paired/domain/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.txt" \
--output_paired_domain_binaryevalformat_only_negative_sentiment_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_negative_sentiment.txt" \
--output_paired_domain_sequence_labels_only_negative_sentiment_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_negative_sentiment.txt" \
--output_paired_domain_binaryevalformat_only_positive_sentiment_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_positive_sentiment.txt" \
--output_paired_domain_sequence_labels_only_positive_sentiment_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_positive_sentiment.txt" \
--output_paired_domain_binaryevalformat_only_negative_sentiment_only_orig_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_negative_sentiment_only_orig.txt" \
--output_paired_domain_sequence_labels_only_negative_sentiment_only_orig_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_negative_sentiment_only_orig.txt" \
--output_paired_domain_binaryevalformat_only_positive_sentiment_only_orig_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_positive_sentiment_only_orig.txt" \
--output_paired_domain_sequence_labels_only_positive_sentiment_only_orig_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_positive_sentiment_only_orig.txt" \
--output_paired_domain_binaryevalformat_only_negative_sentiment_only_new_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_negative_sentiment_only_new.txt" \
--output_paired_domain_sequence_labels_only_negative_sentiment_only_new_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_negative_sentiment_only_new.txt" \
--output_paired_domain_binaryevalformat_only_positive_sentiment_only_new_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.only_positive_sentiment_only_new.txt" \
--output_paired_domain_sequence_labels_only_positive_sentiment_only_new_file "${OUTPUT_DIR}/${INPUT_FILE}.paired.domain_binaryevalformat.sequence_labels.only_positive_sentiment_only_new.txt"


################################################################################
#### Create disjoint train (3.4k and 19k-1.7k+1.7k) and dev sets (dev/2)
#### This allows us to analyze the impact of domain without parallel source/target
#### The size of 3414 comes from the paired combined train data.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/"

OUTPUT_DIR="${DATA_DIR}/combined/paired/disjoint"
mkdir "${OUTPUT_DIR}"


python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/counterfactual_sentiment_data_create_orig_disjoint_splits.py \
--input_train_combined_paired_binaryevalformat_file "${DATA_DIR}/combined/paired/binaryevalformat/train_paired.tsv.binaryevalformat.txt" \
--input_train_full_orig_binaryevalformat_file "${DATA_DIR}/orig/eighty_percent/binaryevalformat/train.tsv.binaryevalformat.txt" \
--final_sample_size 3414 \
--input_dev_combined_paired_binaryevalformat_file "${DATA_DIR}/combined/paired/binaryevalformat/dev_paired.tsv.binaryevalformat.txt" \
--output_train_disjoint_3k_binaryevalformat_file "${OUTPUT_DIR}/train_orig_new_disjoint_3k.tsv.binaryevalformat.txt" \
--output_train_disjoint_full_binaryevalformat_file "${OUTPUT_DIR}/train_orig_new_disjoint_full.tsv.binaryevalformat.txt" \
--output_dev_disjoint_binaryevalformat_file "${OUTPUT_DIR}/dev_orig_new_disjoint.tsv.binaryevalformat.txt"

# Number of filtered lines: 1703
# Length of disjoint_paired_lines: 1711; length of remaining_full_orig_lines: 17475
# Length of train_disjoint_full: 19186
# In the 245 disjoint lines there are: Negative lines: 123; Positive lines: 122


################################################################################
#### Download the Contrast Sets IMDb data
################################################################################

# Note that we most recently checked these pre-processing scripts with a version
# of the data downloaded on 2020-05-05 from
# https://github.com/allenai/contrast-sets/tree/main/IMDb. Note that a more
# recent update of the data fixes some minor formatting issues, but use the older version with
# the following scripts.


################################################################################
#### Preprocess Contrast Sets IMDb data as binaryevalformat
#### The reviews remain one document of sentences.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="UPDATE_WITH_YOUR_PATH/constrast_sets/contrast-sets-master/IMDb/data"

INPUT_DATA_DIR="${DATA_DIR}"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/binaryevalformat"
mkdir "${OUTPUT_DATA_DIR}"

INPUT_FILE="test_original.tsv"

EXPECTED_NUM_FILEDS=2


echo "INPUT_FILE=${INPUT_FILE}"

# we check against the original in the counterfactually-augmented set for test_original.tsv
python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/contrast_sets/contrast_sets_data_to_binaryevalformat.py \
--input_tsv_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--input_counterfactual_original_binaryevalformat_file "UPDATE_WITH_YOUR_PATH/counterfactually-augmented-data-master/sentiment/orig/binaryevalformat/test.tsv.binaryevalformat.txt" \
--expected_number_of_fields ${EXPECTED_NUM_FILEDS} \
--output_binaryevalformat_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.txt" \
--output_monolabels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.monolabels.txt" \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/" \
--bert_model=${BERT_MODEL}

# INPUT_FILE=test_original.tsv
# Negative count: 243; Positive count: 245
# Mean length: 162.69262295081967; std: 65.35677454071636; min: 23, max: 328
# Length of output: 488
# Checking against the original counterfactually augmented file.
# Number of unmatched output lines: 26
# Number of remaining original lines: 26

for INPUT_FILE in "dev_original.tsv" "dev_contrast.tsv" "test_contrast.tsv";
do
  echo "INPUT_FILE=${INPUT_FILE}"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/contrast_sets/contrast_sets_data_to_binaryevalformat.py \
--input_tsv_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--input_counterfactual_original_binaryevalformat_file "" \
--expected_number_of_fields ${EXPECTED_NUM_FILEDS} \
--output_binaryevalformat_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.txt" \
--output_monolabels_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.binaryevalformat.monolabels.txt" \
--bert_cache_dir="UPDATE_WITH_YOUR_PATH/" \
--bert_model=${BERT_MODEL}

done

# INPUT_FILE=dev_original.tsv
# Negative count: 48; Positive count: 52
# Mean length: 161.25; std: 65.17336495839386; min: 43, max: 322
# Length of output: 100
#
# INPUT_FILE=dev_contrast.tsv
# Negative count: 52; Positive count: 48
# Mean length: 165.8; std: 65.48053145783103; min: 44, max: 335
# Length of output: 100
#
# INPUT_FILE=test_contrast.tsv
# Negative count: 245; Positive count: 243
# Mean length: 163.625; std: 65.3761558371948; min: 25, max: 337
# Length of output: 488


################################################################################
#### Create the domain diff data for the CONTRAST SETS DATA
#### The format matches that of the counterfactually-augmented data domain experiments
#### --- i.e., here, label 0 is the original
#### data and 1 is the new data (including for the sequence labels, where the
#### the labels indicate diffs)
####
#### This also creates the sentiment diffs (for orig and new).
################################################################################

# here and elsewhere, we use orig with original, and new with contrast, interchangeably (respectively)
REPO_DIR=UPDATE_WITH_YOUR_PATH
cd ${REPO_DIR}

DATA_DIR="UPDATE_WITH_YOUR_PATH/constrast_sets/contrast-sets-master/IMDb/data/binaryevalformat"

OUTPUT_SENTIMENT_DATA_DIR="${DATA_DIR}"/sentiment_sequence_labels
OUTPUT_DOMAIN_DATA_DIR="${DATA_DIR}"/domain
mkdir "${OUTPUT_SENTIMENT_DATA_DIR}"
mkdir "${OUTPUT_DOMAIN_DATA_DIR}"


SPLIT_NAME="test"
#SPLIT_NAME="dev"

python -u ${REPO_DIR}/code/data/sentiment/counterfactual_dataset/contrast_sets/contrast_sets_sentiment_data_create_sentiment_diffs_and_domain_diffs.py \
--input_orig_binaryevalformat_file "${DATA_DIR}/${SPLIT_NAME}_original.tsv.binaryevalformat.txt" \
--input_new_binaryevalformat_file "${DATA_DIR}/${SPLIT_NAME}_contrast.tsv.binaryevalformat.txt" \
--output_sentiment_sequence_file_prefix "${OUTPUT_SENTIMENT_DATA_DIR}/${SPLIT_NAME}.binaryevalformat" \
--output_domain_file_prefix "${OUTPUT_DOMAIN_DATA_DIR}/${SPLIT_NAME}.binaryevalformat"

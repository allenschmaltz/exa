################################################################################
#### Here are some additional notes on which datasets correspond to the
#### labels used in the paper (arXiv v6) for training.
################################################################################

################################################################################
#### The 3.4k disjoint set:
#### ORIG._{disjoint}+REV. (1.7k+1.7k)
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/disjoint"

TRAIN_DATA_DIR=${DATA_DIR}
TRAIN_DATA_INPUT_NAME="train_orig_new_disjoint_3k.tsv.binaryevalformat.txt"
DEV_DATA_DIR=${DATA_DIR}
DEV_DATA_INPUT_NAME="dev_orig_new_disjoint.tsv.binaryevalformat.txt"

# used in initial training as follows:
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
# --mode "train" \
# --training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
# --dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \

################################################################################
#### The full/largest disjoint set:
#### ORIG._{disjoint}+REV. (19k-1.7k+1.7k)
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/combined/paired/disjoint"

TRAIN_DATA_DIR=${DATA_DIR}
TRAIN_DATA_INPUT_NAME="train_orig_new_disjoint_full.tsv.binaryevalformat.txt"
DEV_DATA_DIR=${DATA_DIR}
DEV_DATA_INPUT_NAME="dev_orig_new_disjoint.tsv.binaryevalformat.txt"

# used in initial training as follows:
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
# --mode "train" \
# --training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
# --dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \

################################################################################
#### The full original set:
#### ORIG. (19k)
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment/orig"

TRAIN_DATA_DIR=${DATA_DIR}/eighty_percent/binaryevalformat
DEV_DATA_DIR=${DATA_DIR}/binaryevalformat
TRAIN_DATA_INPUT_NAME="train.tsv.binaryevalformat.txt"
DEV_DATA_INPUT_NAME="dev.tsv.binaryevalformat.txt"

# used in initial training as follows:
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
# --mode "train" \
# --training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
# --dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \

################################################################################
#### The 3.4k parallel source-target pair set:
#### ORIG.+REV. (1.7k+1.7k)
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

TRAIN_DATA_DIR=${DATA_DIR}/combined/paired/binaryevalformat
DEV_DATA_DIR=${TRAIN_DATA_DIR}
TRAIN_DATA_INPUT_NAME="train_paired.tsv.binaryevalformat.txt"
DEV_DATA_INPUT_NAME="dev_paired.tsv.binaryevalformat.txt"

# used in initial training as follows:
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
# --mode "train" \
# --training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
# --dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \

################################################################################
#### The full original set combined with the 1.7k counterfactually-augmented reviews set:
#### ORIG.+REV. (19k+1.7k)
################################################################################

SERVER_DRIVE_PATH_PREFIX=UPDATE_WITH_YOUR_PATH

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/corpora/additional/counterfactual/download_2020_04_10_reprocessed_2020_04_17/counterfactually-augmented-data-master/sentiment"

TRAIN_DATA_DIR=${DATA_DIR}/orig/eighty_percent/binaryevalformat
TRAIN_DATA_INPUT_NAME="full_orig_train_with_new_train.tsv.binaryevalformat.txt"
DEV_DATA_DIR=${DATA_DIR}/combined/paired/binaryevalformat
DEV_DATA_INPUT_NAME="dev_paired.tsv.binaryevalformat.txt"

# used in initial training as follows:
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
# --mode "train" \
# --training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
# --dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \

################################################################################
#### Dependencies for code from Rei and Søgaard (2018): https://github.com/marekrei/mltagger
################################################################################

# I used the latest commit as of writing, cb356642ac44d59ef8b82df810fd942db695cd44, at https://github.com/marekrei/mltagger.
#
# I ran the code with Tensorflow '1.15.0' with Cuda 10.0.

################################################################################
#### Training of Rei and Søgaard (2018) on this split/version of CONLL2010
################################################################################


cd UPDATE_WITH_YOUR_PATH/mltagger-master

GPU_ID=0
OUTPUT_LOG_FILE=UPDATE_WITH_YOUR_PATH/output/mltagger/conll2010uncertainty_monolabels/train.log.txt

# the config file is in ${REPO_DIR}/replication_scripts/additional_data/conll2010uncertainty/baselines_from_external_models/lstm/ and can be copied over to the mltagger-master conf directory, or another directory

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u experiment.py conf/config_naacl2018_conll10cue_for_new_uncertainty_splits_monolabels.conf > ${OUTPUT_LOG_FILE}

################################################################################
#### Inference using the above model
################################################################################

#### eval test
SPLIT_NAME="test"
EVAL_FILE=UPDATE_WITH_YOUR_PATH/data/zero/conll_2010/uncertainty/conllformat/conll2010uncertainty.conllformat.${SPLIT_NAME}.txt
SAVED_MODEL=UPDATE_WITH_YOUR_PATH/models/mltagger/conll2010uncertainty_monolabels/saved_model.model
OUTPUT_FILE=UPDATE_WITH_YOUR_PATH/output/mltagger/conll2010uncertainty_monolabels/eval_on_${SPLIT_NAME}.log.txt

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u print_output.py ${SAVED_MODEL} ${EVAL_FILE} > ${OUTPUT_FILE}

    # test_cost_sum: 70.92788987362292
    # test_cost_avg: 0.036617392810337074
    # test_sent_count: 1937.0
    # test_sent_predicted: 413.0
    # test_sent_correct: 371.0
    # test_sent_total: 419.0
    # test_sent_p: 0.8983050847457628
    # test_sent_r: 0.8854415274463007
    # test_sent_f: 0.891826923076923
    # test_sent_f05: 0.8957025591501692
    # test_sent_correct_binary: 1847.0
    # test_sent_accuracy_binary: 0.9535363964894166
    # test_tok_0_map: 0.9463647837499733
    # test_tok_0_p: 0.875
    # test_tok_0_r: 0.7343485617597293
    # test_tok_0_f: 0.798528058877645
    # test_tok_0_f05: 0.8427184466019418
    # test_time: 5.187318325042725


# The output file output/mltagger/conll2010uncertainty/eval_on_test.log.txt will contain the predictions.
# For example, here is sentence 779 shown in Table G.2.

# The C   0.0004081726    0.99791396
# BCL6 C  0.0002528429    0.99791396
# gene C  0.00015220046   0.99791396
# encodes C       0.00013443828   0.99791396
# a C     0.0001808703    0.99791396
# 95-kDa C        0.00013494492   0.99791396
# protein C       0.00010454655   0.99791396
# containing C    0.00012299418   0.99791396
# six C   0.00013363361   0.99791396
# C-terminal C    8.416176e-05    0.99791396
# zinc-finger C   9.146333e-05    0.99791396
# motifs C        9.2446804e-05   0.99791396
# and C   0.000120431185  0.99791396
# an C    0.00012406707   0.99791396
# N-terminal C    7.6681376e-05   0.99791396
# POZ C   9.486079e-05    0.99791396
# domain, C       0.0005930364    0.99791396
# suggesting U    0.99969804      0.99791396
# that C  0.010567337     0.99791396
# it C    0.06779271      0.99791396
# may U   0.99966586      0.99791396
# function C      0.0020688474    0.99791396
# as C    0.0027464628    0.99791396
# a C     0.0017173886    0.99791396
# transcription C 0.0001296103    0.99791396
# factor. C       0.00018662214   0.99791396

################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm -- train a K-NN model
#### Note that for the standard zero shot setting, the model does not have
#### access to token labels from training, but does have sentence level labels
#### from training.
#### Training the K-NN model uses 1/2 the conll2010uncertainty dev set, with the other half
#### serving as the dev set. These splits must be created before training.
#### In the Online Appendix, we report results with the main formulation,
#### the distance-weighted K-NN (KNN_dist.). You can uncomment the blocks for MODEL_TYPE="knn"
#### xor MODEL_TYPE="learned_weighting" to alternatively train the equally-weighted
#### K-NN or the constraint-weighted K-NN, respectively. The variable TOP_K
#### determines the number of nearest neighbors (here, K=8) and INIT_TEMP
#### determines the initial value of the temperature parameter. On the datasets considered
#### in this work, we find that a relatively high initial temperature (resulting in a more diffuse
#### distribution for w_k) works well, and we start with INIT_TEMP=10.0 for all
#### KNN_dist. models across datasets and models.
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

# this is the directory for the min-max fine-tuned model
FINE_TUNE_MODE_DIR="${MODEL_DIR}/min_max_fine_tune"

EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/linearexa/learned"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="19"


#SPLIT_NAME="test"
SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/${SPLIT_NAME}_dist_dir"


TOP_K=8

# KNN_dist.
MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# KNN_equal
# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# KNN_const.
# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

# this directory will contain the saved K-NN model file
LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/${MODEL_DIR}/min_max_fine_tune/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "train_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt \
--test_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
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
--do_lower_case \
--exemplar_sentences_database_file ${DATA_DIR}/conll2010uncertainty.${DATABASE_SPLIT_NAME}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_dir="${LINEAR_EXA_MODEL_DIR}" \
--learning_rate 1.0 \
--approximation_type ${APPROX_TYPE} \
--top_k ${TOP_K} \
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--model_temperature_init_value ${INIT_TEMP} \
--model_support_weights_init_values="1.0,-1.0,2.0" \
--model_bias_init_value="0.0" \
--model_gamma_init_value="1.0" \
--database_data_structure_file "${DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--query_train_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-train-split_chunks_ids.pt" \
--query_eval_split_chunk_ids_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_query-eval-split_chunks_ids.pt" \
--restrict_eval_to_query_eval_split_chunk_ids_file >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.train.r1.log.txt



########################### Additional notes
######### Note that we use --use_sentence_level_database_ground_truth since this K-NN only has access to the document-level labels.


######### MODEL_TYPE="maxent" #########
TOP_K=8

MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

# "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.train.r1.log.txt will contain output along the lines of the following (noting that the eval here is on the K-NN dev split, which is about half the size of the full dev set, since the other half is used to learn the 3 parameters):

        # Model (maxent) weights summary (epoch 14):
        #         model.model_bias: -0.940163254737854
        #         model.model_gamma (y_n): 1.494194507598877
        #         model.model_temperature: 10.021345138549805
        #
        # Current max dev sign_flips_to_original_model score: 32 at epoch 14
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~Beginning eval on dev split (epoch 14)~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluating token level flips
        # Evaluating a restricted set of chunk_ids.
        # TopK: Processing distance chunk id 0 of 17.
        # -----------------------------------
        # Total tokens considered in full evaluation: 19824
        # Tokens that exceeded the original model's max length: 1510. A prediction is made for the comparison metrics, but we exclude these tokens from the sign flip analysis since the original model never sees those tokens, for a total of 18314 under consideration.
        # Sentence-level accuracy (based on token-level predictions): 0.9654320987654321 (out of 810)
        # Reference sentence-level accuracy (based on token-level predictions) of the original model: 0.9617283950617284 (out of 810)
        # Total sign flips (relative to original model): 32 out of 18314; As percent accuracy: 0.9982527028502785
        #         Sign flips by sentence (relative to original model): mean: 0.03950617283950617; min: 0; max: 2; std: 0.2129620683466627
        #         Nearest distances of correct sign (relative to original model): mean: 22.44135856628418; min: 3.162277789670043e-05; max: 47.564205169677734; std: 6.673427581787109; total: 18282
        #         Nearest distances of wrong sign (relative to original model): mean: 24.985858917236328; min: 12.172942161560059; max: 36.220645904541016; std: 6.357558727264404; total: 32
        # Total sign flips (relative to ground-truth): 77 out of 18314; As percent accuracy: 0.9957955662334825
        #         Sign flips by sentence (relative to ground-truth): mean: 0.09506172839506173; min: 0; max: 3; std: 0.36460742784057326
        #         Nearest distances of correct sign (relative to ground-truth): mean: 22.44228744506836; min: 3.162277789670043e-05; max: 47.564205169677734; std: 6.674344062805176; total: 18237
        #         Nearest distances of wrong sign (relative to ground-truth): mean: 23.278093338012695; min: 12.112417221069336; max: 42.274070739746094; std: 6.4743733406066895; total: 77
        #         KNN and original model sign match AND *KNN* true ground-truth prediction: mean: 22.438114166259766; min: 3.162277789670043e-05; max: 47.564205169677734; std: 6.673412799835205; total: 18224; Accuracy: 0.9968274805819932
        #         KNN and original model sign DO NOT match AND *KNN* true ground-truth prediction: mean: 28.294498443603516; min: 19.523155212402344; max: 36.220645904541016; std: 5.263708114624023; total: 13; Accuracy: 0.40625
        # Reference total sign flips (relative to ground-truth) of the *original model*: 71 out of 18314; As percent accuracy: 0.9961231844490553
        # Reference random token-level accuracy (19824 tokens): 0.5008071025020178
        # Reference all 1's token-level accuracy (19824 tokens): 0.012409200968523002
        # Reference all 0's token-level accuracy (19824 tokens): 0.987590799031477
        # Reference random token-level accuracy (18314 tokens): 0.49868952713770887
        # Reference all 1's token-level accuracy (18314 tokens): 0.012831713443267445
        # Reference all 0's token-level accuracy (18314 tokens): 0.9871682865567325
        # Random and majority class baselines for approximation to the original model's predictions
        # Reference random token-level accuracy (18314 tokens): 0.49836190892213605
        # Reference all 1's token-level accuracy (18314 tokens): 0.010920607185759528
        # Reference all 0's token-level accuracy (18314 tokens): 0.9890793928142405
        # Time to complete eval: 0.01793290376663208 minutes
        # DETECTION OFFSET: 0.0
        # GENERATED:
        #         Precision: 0.8726415094339622
        #         Recall: 0.7520325203252033
        #         F1: 0.8078602620087336
        #         F0.5: 0.8455210237659962
        #         MCC: 0.8079087490788351
        # Summary scores: Epoch 14: max_sign_flips_to_original_model (train): 50; max_sign_flips_to_original_model (dev): 32
        # Saving epoch 14 as new best max_sign_flips_to_original_model_epoch model with score of 32



################################################################################
#### MM loss: uniCNN+BERT_{base_uncased}+mm  -- eval the K-NN model from above on the full conll2010uncertainty test set
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
EXEMPLAR_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar"
OUTPUT_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/linearexa/learned/eval"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

mkdir "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"


EPOCH="19"


SPLIT_NAME="test"
#SPLIT_NAME="dev"
#SPLIT_NAME="train"
DATABASE_SPLIT_NAME="train"
#DATABASE_SPLIT_NAME="dev"

BERT_MODEL="bert-base-uncased"
#BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
NUM_LAYERS='-1,-2,-3,-4'
#NUM_LAYERS='-1'

DROPOUT_PROB=0.50


DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/${SPLIT_NAME}_dist_dir"

DATABASE_DISTANCE_DEV_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/output/zero_shot_labeling/fce_cnn_interp/${EXPERIMENT_LABEL}/min_max_fine_tune/out/exemplar/dev_dist_dir"


TOP_K=8
MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

# MODEL_TYPE="knn"
# APPROX_TYPE="fce_dev_split"

# MODEL_TYPE="learned_weighting"
# INIT_TEMP=5.0
# APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"

EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"


LINEAR_EXA_MODEL_DIR="${SERVER_SCRATCH_DRIVE_PATH_PREFIX}/${MODEL_DIR}/min_max_fine_tune/linear_exa_${EXA_MODEL_SUFFIX}"
mkdir -p "${LINEAR_EXA_MODEL_DIR}"


CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u exa.py \
--mode "eval_linear_exa" \
--model "non-static" \
--dataset "aesw" \
--word_embeddings_file ${MODEL_DIR}/not_used.txt \
--test_file ${DATA_DIR}/conll2010uncertainty.${SPLIT_NAME}.txt \
--test_seq_labels_file ${DATA_DIR}/conll2010uncertainty.labels.${SPLIT_NAME}.txt \
--max_length 50 \
--max_vocab_size 7500 \
--vocab_file ${MODEL_DIR}/vocab7500k.txt \
--epoch 40 \
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
--do_lower_case \
--exemplar_sentences_database_file ${DATA_DIR}/conll2010uncertainty.${DATABASE_SPLIT_NAME}.txt \
--exemplar_data_database_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${DATABASE_SPLIT_NAME}.exemplar.txt \
--exemplar_data_query_file "${EXEMPLAR_DIR}"/zero_shot.min_max_fine_tune.epoch${EPOCH}.${SPLIT_NAME}.exemplar.txt \
--exemplar_k 1 \
--do_not_apply_relu_on_exemplar_data \
--exemplar_print_type 5 \
--distance_sentence_chunk_size 50 \
--distance_dir "${DISTANCE_DEV_DIR}" \
--save_dir="${LINEAR_EXA_MODEL_DIR}" \
--learning_rate 1.0 \
--approximation_type ${APPROX_TYPE} \
--top_k ${TOP_K} \
--saved_model_file "${LINEAR_EXA_MODEL_DIR}/aesw_non-static_best_max_sign_flips_to_original_model_epoch.pt" \
--use_sentence_level_database_ground_truth \
--max_metric_label "sign_flips_to_original_model" \
--model_type ${MODEL_TYPE} \
--database_data_structure_file "${DATABASE_DISTANCE_DEV_DIR}/database_${DATABASE_SPLIT_NAME}_data_structure.pt" \
--query_data_structure_file "${DISTANCE_DEV_DIR}/query_${SPLIT_NAME}_data_structure.pt" \
--print_error_analysis \
--max_exemplars_to_return ${TOP_K} \
--output_analysis_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.savetype3.r1.txt \
--output_save_type 3 \
--binomial_sample_p 0.01 \
--output_annotations_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.annotations.r1.txt \
--save_annotations \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--save_prediction_stats >"${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.r1.log.txt

######### MODEL_TYPE="maxent" #########
TOP_K=8

MODEL_TYPE="maxent"
INIT_TEMP=10.0
APPROX_TYPE="fce_dev_split_temp${INIT_TEMP}"
EXA_MODEL_SUFFIX="tanh_knn${TOP_K}_${APPROX_TYPE}_model-${MODEL_TYPE}"

# "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.r1.log.txt should contain output along the lines of the following:

      # Total tokens considered in full evaluation: 47614
      # Tokens that exceeded the original model's max length: 3978. A prediction is made for the comparison metrics, but we exclude these tokens from the sign flip analysis since the original model never sees those tokens, for a total of 43636 under consideration.
      # Sentence-level accuracy (based on token-level predictions): 0.9468249870934434 (out of 1937)
      # Reference sentence-level accuracy (based on token-level predictions) of the original model: 0.9483737738771296 (out of 1937)
      # Total sign flips (relative to original model): 95 out of 43636; As percent accuracy: 0.9978228985241544
      #         Sign flips by sentence (relative to original model): mean: 0.049044914816726896; min: 0; max: 3; std: 0.23867290665662283
      #         Nearest distances of correct sign (relative to original model): mean: 23.849138259887695; min: 3.162277789670043e-05; max: 49.6583251953125; std: 7.034792423248291; total: 43541
      #         Nearest distances of wrong sign (relative to original model): mean: 25.310773849487305; min: 9.819195747375488; max: 45.339500427246094; std: 7.931647300720215; total: 95
      # Total sign flips (relative to ground-truth): 206 out of 43636; As percent accuracy: 0.995279127326061
      #         Sign flips by sentence (relative to ground-truth): mean: 0.10635002581311306; min: 0; max: 4; std: 0.39881852792344363
      #         Nearest distances of correct sign (relative to ground-truth): mean: 23.844322204589844; min: 3.162277789670043e-05; max: 49.6583251953125; std: 7.034074783325195; total: 43430
      #         Nearest distances of wrong sign (relative to ground-truth): mean: 25.538156509399414; min: 9.819195747375488; max: 44.303131103515625; std: 7.47885799407959; total: 206
      #         KNN and original model sign match AND *KNN* true ground-truth prediction: mean: 23.842103958129883; min: 3.162277789670043e-05; max: 49.6583251953125; std: 7.03216552734375; total: 43382; Accuracy: 0.9963482694471877
      #         KNN and original model sign DO NOT match AND *KNN* true ground-truth prediction: mean: 25.849166870117188; min: 12.512813568115234; max: 45.339500427246094; std: 8.350196838378906; total: 48; Accuracy: 0.5052631578947369
      # Reference total sign flips (relative to ground-truth) of the *original model*: 207 out of 43636; As percent accuracy: 0.9952562104684206
      # Reference random token-level accuracy (47614 tokens): 0.5050195320703995
      # Reference all 1's token-level accuracy (47614 tokens): 0.012412315705464779
      # Reference all 0's token-level accuracy (47614 tokens): 0.9875876842945353
      # Reference random token-level accuracy (43636 tokens): 0.5026125217710148
      # Reference all 1's token-level accuracy (43636 tokens): 0.012833440278668989
      # Reference all 0's token-level accuracy (43636 tokens): 0.987166559721331
      # Random and majority class baselines for approximation to the original model's predictions
      # Reference random token-level accuracy (43636 tokens): 0.4996562471353928
      # Reference all 1's token-level accuracy (43636 tokens): 0.011023008525071043
      # Reference all 0's token-level accuracy (43636 tokens): 0.988976991474929
      #
      # DETECTION OFFSET: 0.0
      # GENERATED:
      #         Precision: 0.854
      #         Recall: 0.7225042301184433
      #         F1: 0.7827681026581119
      #         F0.5: 0.824006175221922
      #         MCC: 0.7830601900445285

# We can then use the archived file to evaluate using the inference-time decision rules. For example, here is the simple ExA rule from earlier versions of the paper:
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExA"

# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.9080459770114943
#         Recall: 0.6683587140439933
#         F1: 0.769980506822612
#         F0.5: 0.8472758472758473
#         MCC: 0.7767592452475653

# Here is the main ExAG rule used in the paper. Adding the constraint with the true document-level label from the support set is essentially the same as above, since the model is already quite a strong predictor over this data.
python exa_analysis_rules.py \
--output_prediction_stats_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.output_archive.r1.pt \
--analysis_type "ExAG"


# DETECTION OFFSET: 0.0
# GENERATED:
#         Precision: 0.9090909090909091
#         Recall: 0.6598984771573604
#         F1: 0.7647058823529411
#         F0.5: 0.8452535760728218
#         MCC: 0.7722345513100423


# We can also use this archive file to calculate the distance- and magnitude-based constraints, as shown in the more extensive replication scripts of the experiments of the main text.

################################################################################
#### Reading the output visualization files
################################################################################

# Additionally, the K-NN eval script above produces visualized output of the predictions. The --output_annotations_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.annotations.r1.txt file contains the text of the input with each token annotated as follows:
# Tokens with the prefix:
# "*" indicate a ground-truth token-level label (y_n)
# "@" indicate a prediction by the original model (here, for example, uniCNN+BERT_{base_uncased}+mm). These predictions will be the same as those contained in the HTML file --visualization_out_file when running --mode "zero" with the original model.
# "$" indicate a prediction by the K-NN approximation
# "^" indicate a token that was cutoff by the original model (i.e., exceeding --max_length used in initial training)

# We can also print out the exemplars for each token, which will appear in the file provided as the argument to --output_analysis_file. This is what we would show an end-user when they want to look further at the prediction over a particular instance, or a particular token in the instance.

# We provide a few options noted below, but in all cases, the general format is as follows, noting that each block is at the unit of analyis of the token (i.e., all K exemplars are associated with a particular token):
#
# In the follwing example, we replace numeric values with X and show options with [possible output 1|possible output 2].


      # ---------------------------------------------
      # [Sign matches ground truth|Sign flip to ground truth] ; [Sign matches original model|Sign flip to original model]
      # Query sentence X; token index: X; query true token label: [0|1]; query model logit: X, KNN logit: X; Exceeded max length: [True|False]
      # "The query sentence text is here, with the particular token of focus indicated by double brackets: [[]]"
      # Displaying 8 of 8 database exemplars
      # Model info: gamma: 1.494194507598877 bias: -0.940163254737854 new decision boundary: 0.940163254737854
      # ~~~~~~~~~~~~~~~~~~~~~~~~~
      # Exemplar at k=0
      #         Database sentence X; k=0 distance: X; token index: X; database true token label: [0|1]; database model logit: X; database TRUE sentence label: [0|1]
      #         Model term: k=0 n=X w_k=X tanh(f_n=X)=X gamma*(y_n=X)=X w_k*(tanh_f_n+gamma*y_n)=X
      #         "The sentence from the support set corresponding to the exemplar token is here, with the particular token of focus indicated by double brackets: [[]]"
      # ~~~~~~~~~~~~~~~~~~~~~~~~~
      # Exemplar at k=1
      #         Database sentence X; k=1 distance: X; token index: X; database true token label: [0|1]; database model logit: X; database TRUE sentence label: [0|1]
      #         Model term: k=0 n=X w_k=X tanh(f_n=X)=X gamma*(y_n=X)=X w_k*(tanh_f_n+gamma*y_n)=X
      #         "The sentence from the support set corresponding to the exemplar token is here, with the particular token of focus indicated by double brackets: [[]]"
      # ~~~~~~~~~~~~~~~~~~~~~~~~~
      # Exemplar at k=2 ....
      # ... continued for the remaining K

# An example is provided in the next section below.

# This file will get rather large if we print out all 8 exemplars for every token. We include a few options. The primary ones are as follows, and are arguments to --output_save_type:
#                              "0: Save all tokens. Note that the file size will be very large. "
#                              "1: Only save KNN sign flips to original model. "
#                              "2: Only save KNN sign flips to ground-truth. "
#                              "3: Save a sample of all tokens. Inclusion is determined by --binomial_sample_p "
# Note that options 4-8 were for internal dev/debugging and may be removed in subsequent versions.



################################################################################
#### Example of output for 1 token in 1 sentence from test (appears in --output_analysis_file "${OUTPUT_DIR}"/linear_exa.epoch${EPOCH}.database_${DATABASE_SPLIT_NAME}.query_${SPLIT_NAME}.${EXA_MODEL_SUFFIX}.eval.analysis_file.savetype3.r1.txt using the above script.)
# Note: To get the KNN output value we need to add the components from each of the K=8 exemplars and then the K-NN's bias term. So for example in this case, the KNN logit is 1.554031252861023. We add each of the inidividual components from each of the 8 exemplars:
# sum of w_k*(tanh_f_n+gamma*y_n):
# >>> x = 0.16176751255989075+0.2158849537372589+0.24786551296710968+0.2527329623699188+0.26385027170181274+0.4437201917171478+0.4484535753726959+0.4599195122718811+
# and then add the bias, -0.940163254737854:
# >>> x+-0.940163254737854
# 1.5540312379598618
################################################################################


# ---------------------------------------------
# Sign matches ground truth; Sign matches original model
# Query sentence 123; token index: 6; query true token label: 1; query model logit: 50.444732666015625, KNN logit: 1.554031252861023; Exceeded max length: False
# The signaling events mediating this effect [[appeared]] to involve the release of H2O2, since LTB4 failed to induce NF-chi B or NF-IL6 in the presence of the scavenger of H2O2, N-acetyl-L-cysteine.
# Displaying 8 of 8 database exemplars
# Model info: gamma: 1.494194507598877 bias: -0.940163254737854 new decision boundary: 0.940163254737854
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=0
#         Database sentence 140; k=0 distance: 23.272571563720703; token index: 7; database true token label: 1; database model logit: 31.025455474853516; database TRUE sentence label: 1
#         Model term: k=0 n=3360 w_k=0.18439601361751556 tanh(f_n=31.025455474853516)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.4599195122718811
#         NF-kappaB regulatory effect of alpha-lipoate and N-acetylcysteine [[appeared]] to be, at least in part, due to their ability to stabilize elevation of [Ca2+]i following oxidant challenge.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=1
#         Database sentence 14781; k=1 distance: 23.52557373046875; token index: 18; database true token label: 1; database model logit: 55.359031677246094; database TRUE sentence label: 1
#         Model term: k=1 n=357440 w_k=0.17979896068572998 tanh(f_n=55.359031677246094)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.4484535753726959
#         The larger crossreacting protein exhibited an electrophoretic mobility of 80 kDa, was localized in the cell cytosol, and [[appeared]] to be specific for activated lymphocytes since it was not detected in several other human cells including monocytes.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=2
#         Database sentence 9584; k=2 distance: 23.631912231445312; token index: 3; database true token label: 1; database model logit: 28.491477966308594; database TRUE sentence label: 1
#         Model term: k=2 n=235924 w_k=0.17790119349956512 tanh(f_n=28.491477966308594)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.4437201917171478
#         This size discrepancy [[appeared]] to be due to cell-specific translational regulation, since overexpression of a retrovirally transfected SCL gene yielded the higher molecular weight forms in most cell lines (GP+E-86, AT2.5, M1) but only the 22 kDa form in the myeloid cell line, WEHI-3B/D+.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=3
#         Database sentence 9907; k=3 distance: 28.84113121032715; token index: 4; database true token label: 1; database model logit: 17.652912139892578; database TRUE sentence label: 1
#         Model term: k=3 n=242951 w_k=0.10578576475381851 tanh(f_n=17.652912139892578)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.26385027170181274
#         These functions of PAF [[appeared]] to be mediated through the cell surface PAF receptors, as two PAF receptor antagonists, WEB 2086 and L-659,989, blocked both the up-regulation of HB-EGF mRNA and kappa B binding activity induced by PAF.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=4
#         Database sentence 4068; k=4 distance: 29.272533416748047; token index: 17; database true token label: 1; database model logit: 42.96690368652344; database TRUE sentence label: 1
#         Model term: k=4 n=101632 w_k=0.10132848471403122 tanh(f_n=42.96690368652344)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.2527329623699188
#         The intracellular dots seen with the anti-D-mib antibodies were distinct from the Dl- and Ser-positive dots and [[appeared]] to result from background staining (data not shown).
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=5
#         Database sentence 4888; k=5 distance: 29.467418670654297; token index: 13; database true token label: 1; database model logit: 48.34950637817383; database TRUE sentence label: 1
#         Model term: k=5 n=122450 w_k=0.09937697649002075 tanh(f_n=48.34950637817383)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.24786551296710968
#         There was no obvious disorganization of the spindle poles, although some MT bundles [[appeared]] to “splay” laterally, projecting from the pole into the adjacent cytoplasm.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=6
#         Database sentence 11727; k=6 distance: 30.851774215698242; token index: 8; database true token label: 1; database model logit: 26.991230010986328; database TRUE sentence label: 1
#         Model term: k=6 n=285301 w_k=0.08655498176813126 tanh(f_n=26.991230010986328)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.2158849537372589
#         The effects of phorbol ester on MNDA mRNA [[appeared]] to be associated with induced differentiation since inhibiting cell proliferation did not alter the level of MNDA mRNA and cell cycle variation in MNDA mRNA levels were not observed.
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Exemplar at k=7
#         Database sentence 12044; k=7 distance: 33.74378967285156; token index: 6; database true token label: 1; database model logit: 28.57532501220703; database TRUE sentence label: 1
#         Model term: k=7 n=291906 w_k=0.064857617020607 tanh(f_n=28.57532501220703)=1.0 gamma*(y_n=1.0)=1.494194507598877 w_k*(tanh_f_n+gamma*y_n)=0.16176751255989075
#         Cells from these double mutant clones [[appeared]] to invade the brain, typically following fiber tracts, and sometimes induced the formation of trachea (Figure 4E).
# ---------------------------------------------

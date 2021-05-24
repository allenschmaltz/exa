# model types
MODEL_KNN_MAXENT = "maxent"
MODEL_KNN_BASIC = "knn"
MODEL_KNN_LEARNED_WEIGHTING = "learned_weighting"

# these are used in printing the eval output
MODEL_ANALYSIS_SIGN_FLIP_TO_ORIGINAL_MODEL = "Sign flip to original model"
MODEL_ANALYSIS_SIGN_MATCHES_ORIGINAL_MODEL = "Sign matches original model"
MODEL_ANALYSIS_SIGN_FLIP_TO_TRUTH = "Sign flip to ground truth"
MODEL_ANALYSIS_SIGN_MATCHES_TRUTH = "Sign matches ground truth"

# these are the markers for annotating the input data
OUTPUT_ANNOTATION_LABEL_TRUTH = "*"
OUTPUT_ANNOTATION_MODEL_PREDICTION = "@"
OUTPUT_ANNOTATION_KNN_PREDICTION = "$"
OUTPUT_ANNOTATION_MAX_LENGTH = "^"  # tokens that the original model never saw since they exceeded the max length; BERT
                                    # tokenization is taken into account

# used in the analysis script
EXA_RULES_ORIGINAL = "ExA"
EXA_RULES_ORIGINAL_AND_TRUE_SENTENCE_LABEL = "ExAG"
EXA_RULES_ORIGINAL_AND_TRUE_TOKEN_LABEL = "ExAT"
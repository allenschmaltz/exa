PAD_SYM = "$$$PAD$$$"
UNK_SYM = "$$$UNK$$$"
PAD_SYM_ID = 0  # can not change
UNK_SYM_ID = 1  # can not change

INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"

INS_START_SYM_ESCAPED = "$ins@"
INS_END_SYM_ESCAPED = "@ins$"
DEL_START_SYM_ESCAPED = "$del@"
DEL_END_SYM_ESCAPED = "@del$"

# This is the amount of padding before and (where applicable) after each sentence. We keep this constant, but
# it could be eliminated, in principle, for uniCNN, but note that if you do this, some of the pre-processing
# scripts will need to be slightly modified, as they expect padding to be > 0. Unless there's a compelling reason to
# change this, it is better to keep this unchanged at 4, as some non-trivial effort has been taken to make sure
# that padding is correctly handled in evaluating the sequence labeler and the sign evaluation of the K-NN,
# conventions of which have only been tested as of writing with this positive padding value.
PADDING_SIZE = 4

AESW_CLASS_LABELS = [0, 1]

ID_CORRECT = 0  # "negative" class (correct token)
ID_WRONG = 1  # "positive class" (token with error)

FORWARD_TYPE_FEATURE_EXTRACTION = "feature_extraction"
FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS = "generate_exemplar_vectors"
FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION = "sequence_labeling_and_sentence_level_prediction"
FORWARD_TYPE_SEQUENCE_LABELING = "sequence_labeling"
FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION = "sentence_level_prediction"
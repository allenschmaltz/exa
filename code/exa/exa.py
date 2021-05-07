"""
Main entry point

"""
from model import CNN
from model_exa import LinearExA
import utils
import constants
import utils_transformer
import utils_eval
import utils_viz
import utils_viz_sentiment
import utils_exemplar
import utils_linear_exa
import utils_train_linear_exa
import utils_eval_linear_exa
import utils_features
import utils_gen_exemplar_data
import utils_sequence_labeling
import utils_train_main
import utils_classification

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy

import math

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from collections import defaultdict

import torch.nn.functional as F

import codecs


def main():

    parser = argparse.ArgumentParser(description="*Train and Eval Exemplar-based Sequence Classifier and K-NN*")
    parser.add_argument("--mode", default="train",
                        help="TODO UPDATE DOC"
                             "train: train (with eval on dev_file) a model / test at "
                             "the sentence level; "
                             "test saved models; zero performs zero shot labeling; "
                             "seq_labeling_fine_tune performs fine tuning with "
                             "token-level labels, which must be provided")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="aesw", help="available datasets: aesw")
    parser.add_argument("--word_embeddings_file", default="", help="word_embeddings_file")
    parser.add_argument("--training_file", default="", help="training_file")
    parser.add_argument("--dev_file", default="", help="dev_file")
    parser.add_argument("--test_file", default="", help="test_file")
    parser.add_argument("--max_length", default=100, type=int, help="max sentence length (set for training); "
                                                                    "eval sentences are truncated to this length "
                                                                    "at inference time")
    parser.add_argument("--max_vocab_size", default=100000, type=int, help="max vocab size (set for training)")
    parser.add_argument("--vocab_file", default="", help="Vocab file")
    parser.add_argument("--use_existing_vocab_file", default=False, action='store_true',
                        help="Use existing vocab for training. Always true for test.")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--dropout_probability", default=0.5, type=float,
                        help="dropout_probability for training; default is 0.5")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used; -1 for cpu")
    parser.add_argument("--save_dir", default="saved_models", help="save_dir")
    parser.add_argument("--saved_model_file", default="", help="path to existing model (for test mode only)")
    parser.add_argument("--score_vals_file", default="", help="score_vals_file")
    parser.add_argument("--seed_value", default=1776, type=int, help="seed_value")
    parser.add_argument("--data_formatter", default="",
                        help="use 'fce' for fce replication; use 'lowercase' for uncased BERT models")
    parser.add_argument("--word_embeddings_file_in_plaintext", default=False, action='store_true',
                        help="embeddings file is in plain text format")

    parser.add_argument("--filter_widths", default="3,4,5", type=str)
    parser.add_argument("--number_of_filter_maps", default="100,100,100", type=str)

    # for BERT:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--bert_gpu", default=1, type=int,
                        help="the number of gpu to be used for the BERT model; -1 for cpu")

    # for zero-shot labeling:
    parser.add_argument("--color_gradients_file", default="", help="color_gradients_file")
    parser.add_argument("--visualization_out_file", default="", help="visualization_out_file")
    parser.add_argument("--correction_target_comparison_file", default="", help="correction_target_comparison_file")
    parser.add_argument("--output_generated_detection_file", default="", help="output_generated_detection_file")
    parser.add_argument("--detection_offset", type=int, help="detection_offset")
    parser.add_argument("--fce_eval", default=False, action='store_true', help="fce_eval")
    parser.add_argument("--test_seq_labels_file", default="", help="test_seq_labels_file")

    # for fine-tuning labeling
    parser.add_argument("--training_seq_labels_file", default="", help="training_seq_labels_file")
    parser.add_argument("--dev_seq_labels_file", default="", help="dev_seq_labels_file")

    parser.add_argument("--forward_type", default=2, type=int, help="forward_type for sequence training: NOTE THIS"
                                                                    "IS NOT LONGER USED.")
    # For topics -- not currently used -- aggregating words by filter turns out to be not that useful in practice:
    # Some filters do seem to pick up on particular types of features (e.g., mis-spellings), but it's not a
    # particularly actionable observation, since there's not a clear-cut distinction across filters and zero'ing-out
    # or modifying a single filter isn't really practical, given the feature overlap across filters,
    # especially in the tails of the distribution. Use the 'features' approach instead.
    parser.add_argument("--output_topics_file", default="", help="output_topics_file")

    # for input data without tokenized negations: i.e., normal text: can't is can't and not can n't
    parser.add_argument("--input_is_untokenized", default=False, action='store_true',
                        help="for input data without tokenized negations: "
                             "i.e., normal text: 'can't' is 'can't' and not 'can n't'")

    ######## For saving features
    # this next arg was from an earlier version; not currently implemeneted:
    # parser.add_argument("--use_sentence_prediction_for_labeling", default=False, action='store_true',
    #                     help="If provided, positive token label are only considered for sentences
    #                     classified as positive.")
    parser.add_argument("--output_neg_features_file", default="", help="output_neg_features_file")
    parser.add_argument("--output_pos_features_file", default="", help="output_pos_features_file")
    parser.add_argument("--output_neg_sentence_features_file", default="", help="output_neg_sentence_features_file")
    parser.add_argument("--output_pos_sentence_features_file", default="", help="output_pos_sentence_features_file")

    ######## Additional training options
    parser.add_argument("--only_save_best_models", default=False, action='store_true',
                        help="only save model with best acc")
    parser.add_argument("--do_not_allow_padding_to_participate_in_min_max_loss", default=False, action='store_true',
                        help="If supplied, any padding tokens are ignored in the min max loss in min_max_fine_tune().")
    parser.add_argument("--eval_fine_tuning_epoch_based_on_local_to_global_accuracy", default=False,
                        action='store_true',
                        help="If supplied with --only_save_best_models, the best fine-tuning epoch is chosen by the "
                             "local to global accuracy.")

    # additional vizualization options -- primarily for the sentiment datasets
    parser.add_argument("--only_visualize_missed_predictions", default=False, action='store_true',
                        help="only_visualize_missed_predictions")
    parser.add_argument("--only_visualize_correct_predictions", default=False, action='store_true',
                        help="only_visualize_correct_predictions")
    parser.add_argument("--do_not_tune_offset", default=False, action='store_true',
                        help="do_not_tune_offset")

    # for exemplar data:
    parser.add_argument("--output_exemplar_data_file", default="", help="output_exemplar_data_file")
    # for exemplar comparisons:
    parser.add_argument("--exemplar_sentences_database_file", default="",
                        help="exemplar_sentences_database_file (must correspond to exemplar_data_database_file)")
    parser.add_argument("--exemplar_data_database_file", default="", help="exemplar_data_database_file")
    parser.add_argument("--exemplar_data_query_file", default="", help="exemplar_data_database_file")
    parser.add_argument("--do_not_apply_relu_on_exemplar_data", default=False, action='store_true',
                        help="do_not_apply_relu_on_exemplar_data")
    parser.add_argument("--exemplar_k", default=1, type=int, help="NO LONGER USED. Use --top_k")
    # parser.add_argument("--softmax_cutoff", default=1.0, type=float, help="softmax_cutoff")
    parser.add_argument("--exemplar_print_type", default=1, type=int, help="NO LONGER USED. Use --output_save_type")

    ######## The following are for LinearExA (K-NN) model
    parser.add_argument("--distance_sentence_chunk_size", default=50, type=int, help="Number of sentences in an "
                                                                                      "archive of token-level "
                                                                                      "distances for the query.")
    parser.add_argument("--distance_dir", default="", help="distance_dir")
    parser.add_argument("--save_database_data_structure", default=False, action='store_true',
                        help="Multiple query files may have the same database data structures, which only need to "
                             "be saved once.")
    parser.add_argument("--database_data_structure_file", default="", help="database_data_structure_file")
    parser.add_argument("--query_data_structure_file", default="", help="query_data_structure_file")
    parser.add_argument("--create_train_eval_split_from_query_for_knn", default=False, action='store_true',
                        help="Split the query according to --binomial_sample_p by chunk_ids, where "
                             "--binomial_sample_p is the probability to be in the training split, saving the "
                             "train chunk_ids to --query_train_split_chunk_ids_file and the eval chunk_ids to "
                             "--query_eval_split_chunk_ids_file.")
    parser.add_argument("--query_train_split_chunk_ids_file", default="", help="These chunk_ids from the query are "
                                                                               "used to train the KNN.")
    parser.add_argument("--query_eval_split_chunk_ids_file", default="", help="These chunk_ids from the query are "
                                                                               "used to evaluate the KNN.")
    # control the split used for eval
    parser.add_argument("--restrict_eval_to_query_eval_split_chunk_ids_file", default=False, action='store_true',
                        help="If provided, then for --mode 'eval_linear_exa', evaluation is only performed "
                             "on the chunk_ids of --query_eval_split_chunk_ids_file. If provided for "
                             "--mode 'train_linear_exa' --query_train_split_chunk_ids_file is the train split and"
                             "--query_eval_split_chunk_ids_file is the dev split.")

    parser.add_argument("--approximation_type", default="knn", help="approximation_type")
    parser.add_argument("--top_k", default=5, type=int,
                        help="top_k used in training/inference for K-NN. This is also used when caching exemplar "
                             "distances. The top_k for caching can be larger than that used for the subsequent K-NN, "
                             "but must be at least as large.")

    parser.add_argument("--use_sentence_level_database_ground_truth", default=False, action='store_true',
                        help="use_sentence_level_database_ground_truth")
    parser.add_argument("--use_token_level_database_ground_truth", default=False, action='store_true',
                        help="This is the only other option other than --use_sentence_level_database_ground_truth, "
                             "but we include this flag to emphasize that this is only applicable for fully "
                             "supervised sequence models, or in cases where the database is subsequently "
                             "being updated.")
    parser.add_argument("--max_metric_label", default="sign_flips_to_original_model", type=str,
                        help="Determines epoch to save. For the 'pure' zero-shot sequence labeling setting "
                             "for the K-NN, "
                             "this should be 'sign_flips_to_original_model' (default), which is used for all of the "
                             "*K-NN* models in the paper, including the supervised setting. "
                             "For the supervised K-NN setting, this could alternatively be the following (but note "
                             "that the sign mis-match masking in training is currently intended for the "
                             "'sign_flips_to_original_model' option, given the loss is against the original "
                             "sign of the model): "
                             "'precision'; "
                             "'recall'; "
                             "'fscore_1'; "
                             "'fscore_0_5'; "
                             "'mcc'; "
                             "'sign_flips_to_ground_truth'. "
                             "For fine-tuning the supervised sequence-level model, "
                             "the following options are available: "
                             "'precision'; "
                             "'recall'; "
                             "'fscore_1'; "
                             "'fscore_0_5'; "
                             "'mcc'")
    parser.add_argument("--use_sgd", default=False, action='store_true',
                        help="use_sgd")
    parser.add_argument("--model_type", default="knn", help="model_type: knn; maxent; learned_weighting")
    parser.add_argument("--model_temperature_init_value", default=10.0, type=float,
                        help="Initial temperature value. Applicable for model_type: maxent, learned_weighting")
    parser.add_argument("--model_support_weights_init_values", default="1.0,-1.0,2.0", type=str,
                        help="List of reals: 'torch.linspace start, torch.linspace end, starting value for k==0'. "
                             "Applicable for model_type: learned_weighting")
    parser.add_argument("--model_bias_init_value", default=0.0, type=float,
                        help="Initial bias value. Applicable for all model_types.")
    parser.add_argument("--model_gamma_init_value", default=1.0, type=float,
                        help="Initial gamma value (gamma * y_n). Applicable for all model_types.")
    parser.add_argument("--randomize_min_index_limit", default=False, action='store_true',
                        help="randomize_min_index_limit; Applicable for model_type: learned_weighting")
    parser.add_argument("--print_error_analysis", default=False, action='store_true',
                        help="print_error_analysis")
    parser.add_argument("--max_exemplars_to_return", default=10, type=int,
                        help="When printing exemplars, the max number to include in the output. Note that the model "
                             "always calculates output based on K for knn and learned_weighting, but the calculation "
                             "will not be correct (as of this version) for maxent if this value is less than "
                             "the original K. This is for display purposes to make the "
                             "output easier to follow and to reduce the file sizes.")

    parser.add_argument("--output_analysis_file", default="", help="Destination output file for exemplar analysis.")
    parser.add_argument("--output_save_type", default=1, type=int,
                        help="When printing exemplars, the type of token predictions to save: "
                             "0: Save all tokens. Note that the file size will be very large. "
                             "1: Only save KNN sign flips to original model. "
                             "2: Only save KNN sign flips to ground-truth. "
                             "3: Save a sample of all tokens. Inclusion is determined by --binomial_sample_p "
                             "4: Only save distances less than 1.0. "
                             "5: Only save distances less than 1.0 AND indexes within --max_length (padding tokens"
                             "   resolve to 0's, so should be excluded when analyzing the fidelity of the "
                             "   distances) Note that this does not consider BERT tokenizations. "
                             "6: Only save distances less than 1.0 AND query model logit != 0. In these models, "
                             "   query model logit is 0 for padding tokens. "
                             "7: Only save instances for which the query model logit == 0. In these models, "
                             "   query model logit should only be 0 for padding tokens."
                             "8: Only save instances for which the query token mask == 0. These are tokens that were "
                             "   never seen by the original model, since they exceeded the model's original "
                             "   max length."
                        )
    parser.add_argument("--binomial_sample_p", default=0.5, type=float,
                        help="If --output_save_type 3, tokens are saved if "
                             "1==np.random.default_rng(seed=1776).binomial(1, THIS VALUE, size=1)")
    # save the input test file
    parser.add_argument("--save_annotations", default=False, action='store_true', help="save_annotations")
    parser.add_argument("--output_annotations_file", default="", help="Destination output file for the input, "
                                                                      "annotated with ground-truth, model, and KNN "
                                                                      "predictions.")
    parser.add_argument("--never_mask_sign_matches", default=False, action='store_true', help="never_mask_sign_matches "
                                                                                              "during training.")
    parser.add_argument("--never_mask_beyond_max_length_padding", default=False, action='store_true',
                        help="If provided, tokens that exceeded the max length of the original model are NOT masked "
                             "during training. Typically this should NOT be provided, as we typically want to ignore "
                             "any tokens not seen by the original model when training the subsequent K-NN.")
    # save prediction output for analysis
    parser.add_argument("--save_prediction_stats", default=False, action='store_true', help="save_prediction_stats")
    parser.add_argument("--output_prediction_stats_file", default="", help="Destination file for the output "
                                                                           "prediction stats, saved as an archive.")
    # update the database for the LinearExA model before eval
    # the arguments --use_sentence_level_database_ground_truth and --use_token_level_database_ground_truth control
    # the labels used and the database structure arguments should include the new database
    parser.add_argument("--update_knn_model_database", default=False, action='store_true',
                        help="update_knn_model_database")
    # In this case, any database prediction whose sign differs with that of the ground-truth token-level labels
    # is set to 0.
    parser.add_argument("--zero_original_model_logits_not_matching_label_sign", default=False, action='store_true',
                        help="zero_original_model_logits_not_matching_label_sign")

    options = parser.parse_args()

    seed_value = options.seed_value
    max_length = options.max_length
    max_vocab_size = options.max_vocab_size
    vocab_file = options.vocab_file
    use_existing_vocab_file = options.use_existing_vocab_file
    training_file = options.training_file.strip()
    dev_file = options.dev_file.strip()
    test_file = options.test_file.strip()
    data_formatter = options.data_formatter.strip()
    word_embeddings_file_in_plaintext = options.word_embeddings_file_in_plaintext

    torch.manual_seed(seed_value)
    np_random_state = np.random.RandomState(seed_value) # should use this below for the random baseline

    if options.gpu != -1 or options.bert_gpu != -1:
        torch.cuda.manual_seed_all(seed_value)

    main_device = torch.device(f"cuda:{options.gpu}" if options.gpu > -1 else "cpu")

    # for zero-shot labeling:
    color_gradients_file = options.color_gradients_file
    visualization_out_file = options.visualization_out_file
    correction_target_comparison_file = options.correction_target_comparison_file.strip()
    output_generated_detection_file = options.output_generated_detection_file.strip()
    detection_offset = options.detection_offset
    fce_eval = options.fce_eval

    assert options.dataset == "aesw"
    assert options.score_vals_file.strip() != ""

    filter_widths = [int(x) for x in options.filter_widths.split(",")]
    number_of_filter_maps = [int(x) for x in options.number_of_filter_maps.split(",")]
    print(f"CNN: Filter widths: {filter_widths}")
    print(f"CNN: Number of filter maps: {number_of_filter_maps}")

    # sequence-level labels
    training_seq_labels_file = options.training_seq_labels_file.strip()
    if training_seq_labels_file == "":
        training_seq_labels_file = None
    dev_seq_labels_file = options.dev_seq_labels_file.strip()
    if dev_seq_labels_file == "":
        dev_seq_labels_file = None
    test_seq_labels_file = options.test_seq_labels_file.strip()
    if test_seq_labels_file == "":
        test_seq_labels_file = None


    if options.bert_model.strip() != "":
        bert_device = torch.device(f"cuda:{options.bert_gpu}" if options.bert_gpu > -1 else "cpu")
        tokenizer = BertTokenizer.from_pretrained(options.bert_model, do_lower_case=options.do_lower_case,
                                                  cache_dir=options.bert_cache_dir)
        bert_model = BertModel.from_pretrained(options.bert_model, cache_dir=options.bert_cache_dir)
        print("Placing BERT on GPU and setting to eval")
        bert_model.to(bert_device)
        bert_model.eval()

        bert_layers = [int(x) for x in options.bert_layers.split(",")]
        if options.bert_model == "bert-large-cased":
            bert_emb_size = 1024*len(bert_layers)
        elif options.bert_model == "bert-base-cased" or options.bert_model == "bert-base-uncased":
            bert_emb_size = 768*len(bert_layers)
        else:
            assert False, "Not implemented"
    else:
        print("Not using BERT model")
        bert_device = None
        tokenizer = None
        bert_model = None
        bert_layers = None
        bert_emb_size = 0

    if options.mode.strip() == "train":
        print(f"Training mode")
        data = utils.read_aesw_bert(training_file, dev_file, "", None, None, None, data_formatter, tokenizer,
                                    options.input_is_untokenized)
        if use_existing_vocab_file:
            print(f"Using the existing vocab at {vocab_file}")
            vocab, word_to_idx, idx_to_word = utils.load_vocab(vocab_file)
        else:
            vocab, word_to_idx, idx_to_word = utils.get_vocab(data["train_x"], max_vocab_size)
            utils.save_vocab(vocab_file, vocab, word_to_idx)
        data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"] = \
            utils.preprocess_sentences(data["train_x"], word_to_idx, max_length, tokenizer)
        data["idx_dev_x"], data["dev_bert_idx_sentences"], data["dev_bert_input_masks"] = \
            utils.preprocess_sentences(data["dev_x"], word_to_idx, max_length, tokenizer)
    elif options.mode.strip() == "seq_labeling_fine_tune" or options.mode.strip() == "min_max_fine_tune":
        assert training_seq_labels_file is not None and dev_seq_labels_file is not None
        print(f"Sequence-level fine-tuning mode")
        data = utils.read_aesw_bert(training_file, dev_file, "", training_seq_labels_file, dev_seq_labels_file,
                                    None, data_formatter, tokenizer, options.input_is_untokenized)
        print(f"Using the existing vocab at {vocab_file}")
        vocab, word_to_idx, idx_to_word = utils.load_vocab(vocab_file)

        data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"] = \
            utils.preprocess_sentences(data["train_x"], word_to_idx, max_length, tokenizer)
        data["train_padded_seq_y"], data["train_padded_seq_y_mask"] = \
            utils.preprocess_sentences_seq_labels_for_training(data["train_seq_y"], max_length)

        data["idx_dev_x"], data["dev_bert_idx_sentences"], data["dev_bert_input_masks"] = \
            utils.preprocess_sentences(data["dev_x"], word_to_idx, max_length, tokenizer)
        # Note that utils_classification.test_based_on_contributions uses dev_padded_seq_y_mask which does not
        # contain any signal as to the token-level label. In this setting, any values can be given in the sequence
        # labels file (such as all 0's)--the length just needs to match the tokens.
        data["dev_padded_seq_y"], data["dev_padded_seq_y_mask"] = \
            utils.preprocess_sentences_seq_labels_for_training(data["dev_spread_seq_y"], max_length)
    else:
        print(f"Test mode")
        data = utils.read_aesw_bert("", "", test_file, None, None, test_seq_labels_file, data_formatter, tokenizer,
                                    options.input_is_untokenized)
        vocab, word_to_idx, idx_to_word = utils.load_vocab(vocab_file)
        data["idx_test_x"], data["test_bert_idx_sentences"], data["test_bert_input_masks"] = \
            utils.preprocess_sentences(data["test_x"], word_to_idx, max_length, tokenizer)
        # Note that utils_classification.test_based_on_contributions uses test_padded_seq_y_mask which does not
        # contain any signal as to the token-level label. In this setting, any values can be given in the sequence
        # labels file (such as all 0's)--the length just needs to match the tokens.
        if "test_spread_seq_y" in data:
            data["test_padded_seq_y"], data["test_padded_seq_y_mask"] = \
                utils.preprocess_sentences_seq_labels_for_training(data["test_spread_seq_y"], max_length)
    data["vocab"] = vocab

    data["classes"] = constants.AESW_CLASS_LABELS
    data["word_to_idx"] = word_to_idx
    data["idx_to_word"] = idx_to_word

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "max_length": max_length,
        "total_length": max_length + 2*constants.PADDING_SIZE,
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "vocab_size": len(data["vocab"]),  # note this includes padding and unk
        "CLASS_SIZE": len(data["classes"]),
        "padding_idx": constants.PAD_SYM_ID,
        "FILTERS": filter_widths, #[3, 4, 5],
        "FILTER_NUM": number_of_filter_maps, #[100, 100, 100],
        "DROPOUT_PROB": options.dropout_probability, #0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "main_device": main_device,
        "word_embeddings_file": options.word_embeddings_file,
        "save_dir": options.save_dir,
        "score_vals_file": options.score_vals_file.strip(),
        "word_embeddings_file_in_plaintext": word_embeddings_file_in_plaintext,
        "bert_layers": bert_layers,
        "bert_emb_size": bert_emb_size
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("vocab_size:", params["vocab_size"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print(f"Dropout probability: {params['DROPOUT_PROB']}")
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode.strip() == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        utils_train_main.train(data, params, np_random_state, bert_model, bert_device,
                               only_save_best_models=options.only_save_best_models)
        print("=" * 20 + "TRAINING COMPLETED" + "=" * 20)
    elif options.mode.strip() == "seq_labeling_fine_tune" or options.mode.strip() == "min_max_fine_tune":
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        if model.DROPOUT_PROB != params['DROPOUT_PROB']:
            print(f"Note: The dropout probability differs from the original model. Switching from {model.DROPOUT_PROB} "
                  f"(of the original model) to {params['DROPOUT_PROB']} for fine-tuning.")
            model.DROPOUT_PROB = params['DROPOUT_PROB']
        if options.mode.strip() == "seq_labeling_fine_tune":
            print("=" * 20 + "SEQUENCE LABELING FINE-TUNING STARTED" + "=" * 20)
            utils_train_main.seq_labeling_fine_tune(options, model, data, params, np_random_state, bert_model,
                                                    bert_device, only_save_best_models=options.only_save_best_models)
            print("=" * 20 + "SEQUENCE LABELING FINE-TUNING COMPLETED" + "=" * 20)
        elif options.mode.strip() == "min_max_fine_tune":
            print("=" * 20 + "MIN-MAX FINE-TUNING (with standard censoring) STARTED" + "=" * 20)
            utils_train_main.min_max_fine_tune(options, model, data, params, np_random_state, bert_model, bert_device,
                                               only_save_best_models=options.only_save_best_models)
            print("=" * 20 + "MIN-MAX FINE-TUNING (with standard censoring) COMPLETED" + "=" * 20)

    elif options.mode.strip() == "test":
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        test_acc, score_vals = utils_classification.test(data, model, params, bert_model, bert_device, mode="test")
        print("test acc:", test_acc)

        utils.save_lines(options.score_vals_file, score_vals)
    elif options.mode.strip() == "test_by_contributions":
        print(f"Evaluating by test contributions")
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        test_acc, score_vals = \
            utils_classification.test_based_on_contributions(data, model, params, bert_model, bert_device, mode="test")
        print("test acc:", test_acc)

        utils.save_lines(options.score_vals_file, score_vals)
    elif options.mode.strip() == "zero_sentiment" or options.mode.strip() == "zero":
        # the only diff with "zero" is the viz output
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        contribution_tuples_per_sentence, sentence_probs = \
            utils_sequence_labeling.test_seq_labels(data, model, params, bert_model, bert_device,
                                                    mode="test", fce_eval=True)

        # if not fce_eval:
        #     gold_lines = utils.get_target_labels_lines(test_target_comparison_file, convert_to_int=False)
        if options.mode.strip() == "zero_sentiment":
            viz_options = {"only_visualize_missed_predictions": options.only_visualize_missed_predictions,
                           "only_visualize_correct_predictions": options.only_visualize_correct_predictions}

            assert viz_options["only_visualize_missed_predictions"] + \
                   viz_options["only_visualize_correct_predictions"] <= 1
            # generate visualization
            color_gradients_pos_to_neg = utils_viz_sentiment.load_colors(color_gradients_file)
            webpage_template = utils_viz_sentiment.init_webpage_template()
            webpage_template += \
                utils_viz_sentiment.get_visualziation_lines_sentiment(data["test_y"], data["test_sentences"],
                                                                      data["test_seq_y"],
                                                                      contribution_tuples_per_sentence,
                                                                      sentence_probs, color_gradients_pos_to_neg[63],
                                                                      color_gradients_pos_to_neg[0],
                                                                      detection_offset=0.0,
                                                                      fce_eval=True, viz_options=viz_options)
            webpage_template += "</body></html>"
            utils.save_lines(visualization_out_file, [webpage_template])

            utils_eval.calculate_seq_metrics(data["test_seq_y"], contribution_tuples_per_sentence, sentence_probs,
                                             tune_offset=not options.do_not_tune_offset,
                                             print_baselines=True,
                                             output_generated_detection_file=output_generated_detection_file,
                                             numerically_stable=True, fce_eval=True)

        elif options.mode.strip() == "zero":

            # generate visualization
            color_gradients_pos_to_neg = utils_viz.load_colors(color_gradients_file)
            webpage_template = utils_viz.init_webpage_template()
            webpage_template += utils_viz.get_visualziation_lines(data["test_sentences"], data["test_seq_y"],
                                                                  contribution_tuples_per_sentence, sentence_probs,
                                                                  color_gradients_pos_to_neg[63],
                                                                  color_gradients_pos_to_neg[0],
                                                                  detection_offset=0.0, fce_eval=True)
            webpage_template += "</body></html>"
            utils.save_lines(visualization_out_file, [webpage_template])

            utils_eval.calculate_seq_metrics(data["test_seq_y"], contribution_tuples_per_sentence, sentence_probs,
                                             tune_offset=True,
                                             print_baselines=True,
                                             output_generated_detection_file=output_generated_detection_file,
                                             numerically_stable=True, fce_eval=True)

    elif options.mode.strip() == "features":
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        neg_features_output_lines, pos_features_output_lines, \
            neg_sentence_features_output_lines, pos_sentence_features_output_lines = \
            utils_features.test_seq_labels_features(data, model, params, bert_model, bert_device,
                                                    mode="test", fce_eval=True)

        utils.save_lines(options.output_neg_features_file, neg_features_output_lines)
        utils.save_lines(options.output_pos_features_file, pos_features_output_lines)

        utils.save_lines(options.output_neg_sentence_features_file, neg_sentence_features_output_lines)
        utils.save_lines(options.output_pos_sentence_features_file, pos_sentence_features_output_lines)

    elif options.mode.strip() == "generate_exemplar_data":  # this will save the vectors as a txt file
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        # Note this is from my original code from 2019, and saves the exemplars to a txt file.
        # I've since updated the training of the K-NN to save to pytorch archived tensor chunks.
        # In the queue is to update this initial step, as well.
        contribution_tuples_per_sentence, sentence_probs = \
            utils_gen_exemplar_data.generate_exemplar_data(options.output_exemplar_data_file, data, model,
                                                           params, bert_model, bert_device, mode="test", fce_eval=True)
    # elif options.mode.strip() == "eval_exemplar_auditing_inference_time_decision_rules":
    # In the current version, I've left out the code for just running the inference-time decision rules without the
    # K-NN to keep things simple. In this version, all analyses need to go through the K-NN data structures, and the
    # K-NN analysis script runs the ExA rules for comparison for the paper. (Note that the ExA rules are independent
    # of the particular K-NN or K-NN weights, since they only need the nearest match which is the same regardless of
    # the K-NN, so you can just init a dummy K-NN and run eval if you only need the ExA output. The only caveat
    # is that in the current version, you must then still pre-cache all of the distances.)
    elif options.mode.strip() == "save_exemplar_distances":
        # Here we cache the L2 distances (via exact search) between a query and the database once,
        # since they will not change when training the K-NN. In this version, we only save the top-k distances
        # and indexes into the support set. The command line argument --top_k controls the number of distances
        # to be saved. The archive can contain more distances than the final K-NN, but obviously it needs at least
        # as many. These files are very large if all distances are saved, but in practice, we find that K can
        # be relatively small and still closely approximate the original model's prediction logit. Alternatively (not
        # implemented here), we could dynamically generate the distances during K-NN training/inference if the files are
        # too large to save to disk all at once. See the Memory Matching code base for a possible approach, where
        # I iteratively save the distances to disk. However, importantly, note that the use case of the
        # K-NN for interpretability/explainability is for an
        # already trained model, so the distances themselves never change during training, unlike the
        # retrieval-classification setting, which also has interpretability/explainability properties, but
        # in a different context and use. (I.e., we can also build a K-NN, as shown here, for an already trained
        # retrieval-classification model.)
        # Also, I'm currently saving the floats, but it's probably sufficient to save the distances as fp16 (or
        # possibly, even quantized).
        utils_linear_exa.save_exemplar_distances(params, options, data, number_of_filter_maps)
        # also save the associated data structures
        if options.save_database_data_structure:
            # optionally, save database data structures -- only need to do this once (but it must be done for each
            # database)
            database_data_structures = utils_eval_linear_exa.init_database_data_structures(data, options,
                                                                                           number_of_filter_maps)
            utils_linear_exa.save_data_structure_torch_to_file(options.database_data_structure_file,
                                                               database_data_structures)
        # save query data structures
        eval_data_structures = utils_eval_linear_exa.init_eval_data_structures(data, options, number_of_filter_maps)
        utils_linear_exa.save_data_structure_torch_to_file(options.query_data_structure_file,
                                                           eval_data_structures)
        # optionally, create splits by chunk_ids for "training" and "dev/eval" splits for training a K-NN
        if options.create_train_eval_split_from_query_for_knn:
            # load the query chunk_ids
            chunk_ids = utils_linear_exa.load_memory_structure_torch(options.distance_dir,
                                                                     f"distances_to_db_chunk_ids", 0, -1).numpy()
            rng = np.random.default_rng(seed=options.seed_value)
            query_train_chunk_ids = []
            query_eval_chunk_ids = []
            for chunk_id in chunk_ids:
                if 1 == rng.binomial(1, options.binomial_sample_p, size=1):
                    query_train_chunk_ids.append(chunk_id)
                else:
                    query_eval_chunk_ids.append(chunk_id)
            print(f"Number of chunks in the query-train split: {len(query_train_chunk_ids)}")
            print(f"Number of chunks in the query-eval split: {len(query_eval_chunk_ids)}")
            utils_linear_exa.save_data_structure_torch_to_file(options.query_train_split_chunk_ids_file,
                                                               torch.LongTensor(query_train_chunk_ids))
            utils_linear_exa.save_data_structure_torch_to_file(options.query_eval_split_chunk_ids_file,
                                                               torch.LongTensor(query_eval_chunk_ids))

    elif options.mode.strip() == "train_linear_exa":
        assert not options.print_error_analysis and not options.save_annotations, \
            f"ERROR: Currently saving the output is not allowed in training. Use eval mode instead."
        utils_train_linear_exa.train_linear_exa(params, options, data, number_of_filter_maps,
                                                output_generated_detection_file, np_random_state)

    elif options.mode.strip() == "eval_linear_exa":
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        utils_train_linear_exa.display_model_weights_summary(model, f"Loaded model for eval", options)

        if options.print_error_analysis or options.update_knn_model_database:
            database_data_structures = \
                utils_linear_exa.load_data_structure_torch_from_file(options.database_data_structure_file, -1)
            print(f"Loaded database data structure file from {options.database_data_structure_file}")
        else:
            database_data_structures = None
        eval_data_structures = \
            utils_linear_exa.load_data_structure_torch_from_file(options.query_data_structure_file, -1)
        print(f"Loaded eval data structure file from {options.query_data_structure_file}")

        if options.restrict_eval_to_query_eval_split_chunk_ids_file:
            eval_split_chunk_ids_subset = \
                utils_linear_exa.load_data_structure_torch_from_file(options.query_eval_split_chunk_ids_file, -1).numpy()
            print(f"Evaluating on the subset of data specified by {options.query_eval_split_chunk_ids_file}")
        else:
            print(f"Evaluating on the full query dataset.")
            eval_split_chunk_ids_subset = None

        if options.update_knn_model_database:
            model = utils_train_linear_exa.update_database_of_existing_model(model, params, options,
                                                                             database_data_structures)
        utils_eval_linear_exa.eval_linear_exa(eval_data_structures, model, params, options, data, number_of_filter_maps,
                                              output_generated_detection_file,
                                              database_data_structures=database_data_structures,
                                              eval_split_chunk_ids_subset=eval_split_chunk_ids_subset)


if __name__ == "__main__":
    main()

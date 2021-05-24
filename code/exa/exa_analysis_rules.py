"""
Calculate the exemplar auditing decision rules from the saved K-NN data structures.

Note that such inference-time decision rules are not dependent on the particular K-NN, but we use the K-NN data
structures to access the nearest matches and associated meta-data from the database. This is intended to be simple to
also serve as an example of accessing the data structures; see the additional exa_analysis_[X].py scripts for
additional analyses.
"""

import utils
import constants
import utils_eval
import utils_viz
import utils_exemplar
import utils_linear_exa
import utils_train_linear_exa
import utils_eval_linear_exa
import constants_exa

import torch
import torch.nn as nn

import numpy as np
import argparse
import copy

import math

from collections import defaultdict

import codecs


def score_output(prediction_stats, options):
    print(f"Number of sentences under consideration for analysis: {len(prediction_stats)}")
    test_seq_y_for_eval = []
    sentence_probs = []
    all_contribution_tuples_by_sentence = []

    for archive_index in range(len(prediction_stats)):
        # first read data from archive
        prediction_stats_for_sent = prediction_stats[archive_index]
        sent_i = prediction_stats_for_sent["sent_i"]
        true_sentence_level_label = prediction_stats_for_sent["true_sentence_level_label"]
        test_seq_y_for_sent_i = prediction_stats_for_sent["test_seq_y_for_sent_i"]
        sentence_probs_for_sent = prediction_stats_for_sent["sentence_probs_for_sent_i"]
        knn_bias = prediction_stats_for_sent["knn_bias"]
        token_level_stats_for_all_tokens_in_sentence = prediction_stats_for_sent["token_level_stats"]

        contribution_tuples_for_sent = []
        for token_level_stats in token_level_stats_for_all_tokens_in_sentence:
            named_token_stats = utils_eval_linear_exa.get_token_level_stats_from_data_structure(token_level_stats)

            token_id = named_token_stats["token_id"]
            query_true_token_label = named_token_stats["query_true_token_label"]
            original_model_logit = named_token_stats["original_model_logit"]
            knn_logit = named_token_stats["knn_logit"]
            query_non_padding_mask_for_token = named_token_stats["query_non_padding_mask_for_token"]
            # The following are all lists of size K:
            exemplar_weights_list = named_token_stats["exemplar_weights_list"]
            db_indexes = named_token_stats["db_indexes"]
            db_distances = named_token_stats["db_distances"]
            db_true_token_labels = named_token_stats["db_true_token_labels"]
            db_true_sentence_labels = named_token_stats["db_true_sentence_labels"]
            db_original_model_logits = named_token_stats["db_original_model_logits"]

            # original model TP ExA rule
            if options.analysis_type == constants_exa.EXA_RULES_ORIGINAL:
                if original_model_logit > 0 and db_original_model_logits[0] > 0:
                    contribution_tuples_for_sent.append((0.0, original_model_logit, 0.0, 0.0))
                else:
                    contribution_tuples_for_sent.append((0.0, 0.0, 0.0, 0.0))
            elif options.analysis_type == constants_exa.EXA_RULES_ORIGINAL_AND_TRUE_SENTENCE_LABEL:
                # Remember, we can't use the true token-level labels in the database (e.g., db_true_token_labels[0])
                # in the zero-shot setting, but we can use the *sentence-level* labels in the database
                if original_model_logit > 0 and db_original_model_logits[0] > 0 and db_true_sentence_labels[0] == 1:
                    contribution_tuples_for_sent.append((0.0, original_model_logit, 0.0, 0.0))
                else:
                    contribution_tuples_for_sent.append((0.0, 0.0, 0.0, 0.0))
            elif options.analysis_type == constants_exa.EXA_RULES_ORIGINAL_AND_TRUE_TOKEN_LABEL:
                if original_model_logit > 0 and db_original_model_logits[0] > 0 and db_true_token_labels[0] == 1:
                    contribution_tuples_for_sent.append((0.0, original_model_logit, 0.0, 0.0))
                else:
                    contribution_tuples_for_sent.append((0.0, 0.0, 0.0, 0.0))
            elif options.analysis_type == "KNN":
                contribution_tuples_for_sent.append((0.0, knn_logit, 0.0, 0.0))
            elif options.analysis_type == "original_model":
                contribution_tuples_for_sent.append((0.0, original_model_logit, 0.0, 0.0))
            else:
                assert False

        test_seq_y_for_eval.append(test_seq_y_for_sent_i)
        sentence_probs.append(sentence_probs_for_sent)
        all_contribution_tuples_by_sentence.append(contribution_tuples_for_sent)

    eval_stats = utils_eval.calculate_seq_metrics(test_seq_y_for_eval, all_contribution_tuples_by_sentence,
                                                  sentence_probs, tune_offset=False,
                                                  print_baselines=False,
                                                  output_generated_detection_file="",
                                                  numerically_stable=True, fce_eval=True)


def main():
    parser = argparse.ArgumentParser(description="-----[Analysis]-----")
    parser.add_argument("--output_prediction_stats_file", default="", help="Destination file for the output "
                                                                           "prediction stats, saved as an archive.")
    parser.add_argument("--analysis_type", default="exa", type=str,
                        help="ExA: Original ExA decision rule: Original model token logit > 0 AND "
                             "      k=0 database model token logit > 0,"
                             "      then original model token logit; "
                             "          else 0. "
                             "ExAG: Original ExAG decision rule: Original model token logit > 0 AND "
                             "      k=0 database model token logit > 0 AND"
                             "      k=0 database model sentence true label == 1, "
                             "      then original model token logit; "
                             "          else 0. "
                             "ExAT: Original ExAT decision rule: Original model token logit > 0 AND "
                             "      k=0 database model token logit > 0 AND"
                             "      k=0 database model *token* true label == 1, "
                             "      then original model token logit; "
                             "          else 0. "
                             "KNN: Evaluate the K-NN logit. This should match the standard eval script and is just"
                             "      used to check that the output from the data structures match."
                             "original_model: Evaluate the original model's logit. This should match the standard "
                             "      eval script and is just used to check that the output from the data structures "
                             "      match.")

    options = parser.parse_args()
    prediction_stats = \
                    utils_linear_exa.load_data_structure_torch_from_file(options.output_prediction_stats_file, -1)
    print(f"Loaded prediction stats file from {options.output_prediction_stats_file}")

    print(f"{''.join(['-']*45)}Analysis Type: {options.analysis_type}{''.join(['-']*45)}")
    score_output(prediction_stats, options)

    if options.analysis_type == constants_exa.EXA_RULES_ORIGINAL_AND_TRUE_TOKEN_LABEL:
        print(f"---WARNING---WARNING---WARNING---")
        print(f"The ExAT inference-time decision rule is being used which access the *token-level* labels in "
              f"the database. This is only applicable for the fully-supervised setting, or cases in which the "
              f"database has been updated with token-level labels. For standard sentence-level classification "
              f"settings, use ExAG instead, which only accesses the sentence-level labels in the database.")
    elif options.analysis_type == constants_exa.EXA_RULES_ORIGINAL:
        print(f"---NOTE---")
        print(f"The ExA inference-time decision rule is primarily for analysis and reference purposes "
              f"since it only accesses the model's logit in the database, but it does not access the "
              f"associated ground-truth labels. Typically in practice, at least ExAG would be used since the "
              f"sentence-level labels would typically be available from training. Of course, if the model is already "
              f"a relatively strong classifier over the sentences in the database, then in practice there might "
              f"not be much difference between ExA and ExAG.")


if __name__ == "__main__":
    main()

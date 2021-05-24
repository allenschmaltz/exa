"""
Calculate summary statistics with respect to the approximation, and optionally, the ground-truth. We can then
subsequently use thresholds on the magnitudes of the output and/or distances to the first match to constrain the
output on domain-shifted and out-of-domain data, restricting to the subset of the data over which the model is
most likely to be reliable.

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

    num_max_length_padding_tokens_ignored = 0

    stats = {}

    for label in ["True", "False"]:
        for class_type in ["0", "1"]:
            for analysis_type in ["approximation", "prediction"]:
                # output 'logits'
                stats[f"knn_logit_{analysis_type}{label}_class{class_type}"] = []
                stats[f"original_model_logit_{analysis_type}{label}_class{class_type}"] = []
                # L^2 distances to first match
                stats[f"knn_logit_{analysis_type}{label}_class{class_type}_distance"] = []
                stats[f"original_model_logit_{analysis_type}{label}_class{class_type}_distance"] = []

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

            # For reference we calculate the token-level results using the K-NN or the original model output. This
            # is a useful check to ensure that the correct/expected input archive file is being used.
            # However, use the exa_analysis_rules.py script for the inference-time decision rules.
            if options.analysis_type == "KNN":
                contribution_tuples_for_sent.append((0.0, knn_logit, 0.0, 0.0))
            elif options.analysis_type == "original_model":
                contribution_tuples_for_sent.append((0.0, original_model_logit, 0.0, 0.0))
            else:
                assert False

            include_token = True
            if not options.do_not_exclude_padding and query_non_padding_mask_for_token == 0:
                include_token = False
                num_max_length_padding_tokens_ignored += 1
            if include_token:

                approximation_is_correct = (knn_logit > 0 and original_model_logit > 0) or \
                                           (knn_logit <= 0 and original_model_logit <= 0)

                knn_prediction_class = 1 if knn_logit > 0 else 0
                original_model_prediction_class = 1 if original_model_logit > 0 else 0

                # Note that here we do not restrict by first_match_is_knn_tp_or_tn in order to compare to the
                # raw original model output -- which in standard settings wouldn't have access to the KNN. In any
                # case, it's a bit difficult to directly compare to that setting without involving the
                # true token labels, which is a much different setting (i.e., one might as well then use
                # those labels more directly, and conversely, with only the original model, the K-NN approximation
                # isn't available) but here we're
                # just aiming to see if the KNN's output is at least as good a comparator, conditional on the
                # approximation correctness, since in most classification settings, true token-level labels will
                # not be available. (Incidentally, it's very difficult in general to compare reliability/uncertainty
                # measures, when the measure is an integral part of the model itself.)
                stats[f"knn_logit_approximation{approximation_is_correct}_class{knn_prediction_class}"].append(knn_logit)
                stats[f"original_model_logit_approximation{approximation_is_correct}_class{original_model_prediction_class}"].append(original_model_logit)

                # On the other hand, the distances to the first match primarily only make sense as a constraint if
                # the first match is consistent with the overall prediction. In practice, we can combine with the
                # above. Other variations on this theme are possible, but we aim to keep it straightforward here. Note
                # that for the purposes of the analysis relevant for the section of the paper, we are primarily
                # concerned with this w.r.t. to the K-NN, but we also add a similar constraint to the original model
                # stats, as well, below for reference.
                first_match_is_knn_tp_or_tn = (knn_logit > 0 and db_true_sentence_labels[0] == 1) or \
                                              (knn_logit <= 0 and db_true_sentence_labels[0] == 0)
                if first_match_is_knn_tp_or_tn:
                    stats[f"knn_logit_approximation{approximation_is_correct}_class{knn_prediction_class}_distance"].append(db_distances[0])
                # We do not use this directly in the paper, but this is the analogoue for the original model. Note this
                # is slightly weaker than ExAG, since we do not require a match of db_original_model_logits[0]. This is
                # to match the case above with the K-NN, where only considering the first db logit makes less sense.
                first_match_is_original_model_tp_or_tn = (original_model_logit > 0 and db_true_sentence_labels[0] == 1) or \
                                                         (original_model_logit <= 0 and db_true_sentence_labels[0] == 0)
                if first_match_is_original_model_tp_or_tn:
                    stats[f"original_model_logit_approximation{approximation_is_correct}_class{original_model_prediction_class}_distance"].append(db_distances[0])

                if options.output_metrics_against_ground_truth:
                    knn_prediction_is_correct = (knn_logit > 0 and query_true_token_label == 1) or \
                                                (knn_logit <= 0 and query_true_token_label == 0)
                    original_model_prediction_is_correct = (original_model_logit > 0 and query_true_token_label == 1) or \
                                                           (original_model_logit <= 0 and query_true_token_label == 0)
                    stats[f"knn_logit_prediction{knn_prediction_is_correct}_class{knn_prediction_class}"].append(
                        knn_logit)
                    stats[
                        f"original_model_logit_prediction{original_model_prediction_is_correct}_class{original_model_prediction_class}"].append(
                        original_model_logit)

                    if first_match_is_knn_tp_or_tn:
                        stats[
                            f"knn_logit_prediction{knn_prediction_is_correct}_class{knn_prediction_class}_distance"].append(
                            db_distances[0])
                    if first_match_is_original_model_tp_or_tn:
                        stats[
                            f"original_model_logit_prediction{original_model_prediction_is_correct}_class{original_model_prediction_class}_distance"].append(
                            db_distances[0])

        test_seq_y_for_eval.append(test_seq_y_for_sent_i)
        sentence_probs.append(sentence_probs_for_sent)
        all_contribution_tuples_by_sentence.append(contribution_tuples_for_sent)

    # Note that this is only run as a check to make sure the correct input files are chosen. This should match the
    # output from the main scripts.
    eval_stats = utils_eval.calculate_seq_metrics(test_seq_y_for_eval, all_contribution_tuples_by_sentence,
                                                  sentence_probs, tune_offset=False,
                                                  print_baselines=False,
                                                  output_generated_detection_file="",
                                                  numerically_stable=True, fce_eval=True)

    if not options.do_not_exclude_padding:
        print(f"Number of max length padding tokens ignored in the following analyses: "
              f"{num_max_length_padding_tokens_ignored}")

    print(f"{''.join(['-'] * 45)}Approximation Summary Stats{''.join(['-'] * 45)}")
    for label in ["True", "False"]:
        print(f"--------Approximation is correct: {label}--------")
        for class_type in ["0", "1"]:
            print(f"*Class: {class_type}*")
            print(f"K-NN output:")
            print_summary_stats(f"knn_logit_approximation{label}_class{class_type}",
                                stats[f"knn_logit_approximation{label}_class{class_type}"])
            print(f"K-NN distance:")
            print_summary_stats(f"knn_logit_approximation{label}_class{class_type}_distance",
                                stats[f"knn_logit_approximation{label}_class{class_type}_distance"])
        print(f"*Class: 0 and 1*")
        print(f"K-NN distance (both classes combined):")
        print_summary_stats(f"knn_logit_approximation{label}_class0AND1_distance",
                            stats[f"knn_logit_approximation{label}_class{0}_distance"]+
                            stats[f"knn_logit_approximation{label}_class{1}_distance"])
        for class_type in ["0", "1"]:
            print(f"*Class: {class_type}*")
            print(f"Original Model output:")
            print_summary_stats(f"original_model_logit_approximation{label}_class{class_type}",
                                stats[f"original_model_logit_approximation{label}_class{class_type}"])
            print(f"Original Model distance:")
            print_summary_stats(f"original_model_logit_approximation{label}_class{class_type}_distance",
                                stats[f"original_model_logit_approximation{label}_class{class_type}_distance"])
        print(f"*Class: 0 and 1*")
        print(f"Original Model distance (both classes combined):")
        print_summary_stats(f"original_model_logit_approximation{label}_class0AND1_distance",
                            stats[f"original_model_logit_approximation{label}_class{0}_distance"]+
                            stats[f"original_model_logit_approximation{label}_class{1}_distance"])
    if options.output_metrics_against_ground_truth:
        print(f"{''.join(['-'] * 45)}Ground-truth Summary Stats{''.join(['-'] * 45)}")
        for label in ["True", "False"]:
            print(f"--------Prediction is correct: {label}--------")
            for class_type in ["0", "1"]:
                print(f"*Class: {class_type}*")
                print(f"K-NN output:")
                print_summary_stats(f"knn_logit_prediction{label}_class{class_type}",
                                    stats[f"knn_logit_prediction{label}_class{class_type}"])
                print(f"K-NN distance:")
                print_summary_stats(f"knn_logit_prediction{label}_class{class_type}_distance",
                                    stats[f"knn_logit_prediction{label}_class{class_type}_distance"])
            print(f"*Class: 0 and 1*")
            print(f"K-NN distance (both classes combined):")
            print_summary_stats(f"knn_logit_prediction{label}_class0AND1_distance",
                                stats[f"knn_logit_prediction{label}_class{0}_distance"] +
                                stats[f"knn_logit_prediction{label}_class{1}_distance"])
            for class_type in ["0", "1"]:
                print(f"*Class: {class_type}*")
                print(f"Original Model output:")
                print_summary_stats(f"original_model_logit_prediction{label}_class{class_type}",
                                    stats[f"original_model_logit_prediction{label}_class{class_type}"])
                print(f"Original Model distance:")
                print_summary_stats(f"original_model_logit_prediction{label}_class{class_type}_distance",
                                    stats[f"original_model_logit_prediction{label}_class{class_type}_distance"])
            print(f"*Class: 0 and 1*")
            print(f"Original Model distance (both classes combined):")
            print_summary_stats(f"original_model_logit_prediction{label}_class0AND1_distance",
                                stats[f"original_model_logit_prediction{label}_class{0}_distance"]+
                                stats[f"original_model_logit_prediction{label}_class{1}_distance"])


def print_summary_stats(label, list_of_floats):
    print(f"\tLabel: {label}")
    if len(list_of_floats) > 0:
        print(f"\t\t"
              f"mean: {np.mean(list_of_floats)}; "
              f"min: {np.min(list_of_floats)}; "
              f"max: {np.max(list_of_floats)}; "
              f"std: {np.std(list_of_floats)}; "
              f"total: {len(list_of_floats)}")
    else:
        print(f"\t\tEmpty list")


def main():
    parser = argparse.ArgumentParser(description="-----[Analysis]-----")
    parser.add_argument("--output_prediction_stats_file", default="", help="Destination file for the output "
                                                                           "prediction stats, saved as an archive.")
    parser.add_argument("--analysis_type", default="exa", type=str,
                        help="KNN: Evaluate the K-NN logit. This should match the standard eval script and is just"
                             "      used to check that the output from the data structures match."
                             "original_model: Evaluate the original model's logit. This should match the standard "
                             "      eval script and is just used to check that the output from the data structures "
                             "      match.")
    parser.add_argument("--do_not_exclude_padding", default=False, action='store_true',
                        help="Typically this should NOT be provided, as we want to analyze the approximations "
                             "only in terms of tokens that the original model actually saw (i.e., we want "
                             "to exclude padding).")
    parser.add_argument("--output_metrics_against_ground_truth", default=False, action='store_true',
                        help="If included, comparisons against the ground-truth are generated. Otherwise, only "
                             "results relative to the approximations are generated (i.e., whether the output of "
                             "the K-NN and the original model match).")

    options = parser.parse_args()

    prediction_stats = \
                    utils_linear_exa.load_data_structure_torch_from_file(options.output_prediction_stats_file, -1)
    print(f"Loaded prediction stats file from {options.output_prediction_stats_file}")

    print(f"{''.join(['-']*45)}Analysis Type: {options.analysis_type}{''.join(['-']*45)}")
    score_output(prediction_stats, options)

    # Note that here we're somewhat loose with the term 'logit', which in the use here just means the
    # untransformed output of the original network in [-inf, inf], or the bounded value of the K-NNs in
    # [-2+constant, 2+constant].
    print(f"All results here are with a decision boundary of 0")
    if options.do_not_exclude_padding:
        print(f"WARNING: Padding tokens are included in the metrics. These tokens were never seen by the original "
              f"model.")
    else:
        print(f"In these summary stats, padding tokens have been excluded. These tokens were never seen by the "
              f"original model.")


if __name__ == "__main__":
    main()

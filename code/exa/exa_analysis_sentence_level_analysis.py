"""
Calculate sentence-level accuracy with supplied constraints. The constraints can be determined from the
dev set using exa_analysis_sentence_level_analysis_summary_stats.py.

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

    output_prediction_tuples = []

    sentence_level_y = []
    sentence_level_prediction = []

    sentence_level_y_constrained = []
    sentence_level_prediction_constrained = []

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

        sentence_level_y.append(true_sentence_level_label)
        sentence_level_prediction.append(1.0 if sentence_probs_for_sent[1]>sentence_probs_for_sent[0] else 0.0)

        admit_constraints = []
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

            if options.constrain_by_knn_output_magnitude:
                if (knn_logit > 0 and np.abs(knn_logit) > options.class1_magnitude_threshold) or \
                   (knn_logit <= 0 and np.abs(knn_logit) > options.class0_magnitude_threshold):
                    admit_via_threshold_constraints = True
                else:
                    admit_via_threshold_constraints = False
            elif options.constrain_by_original_model_output_magnitude:
                if (original_model_logit > 0 and np.abs(original_model_logit) > options.class1_magnitude_threshold) or \
                   (original_model_logit <= 0 and np.abs(original_model_logit) > options.class0_magnitude_threshold):
                    admit_via_threshold_constraints = True
                else:
                    admit_via_threshold_constraints = False
            else:
                admit_via_threshold_constraints = True
            admit_token = 1
            if admit_via_threshold_constraints:
                if not options.constrain_by_nearest_distance:
                    output_prediction_tuples.append(
                        (query_true_token_label, original_model_logit, np.abs(original_model_logit),
                         knn_logit, np.abs(knn_logit), db_distances[0]))
                else:  # [and/or] consider distance constraint
                    if options.nearest_distance_constraint_is_relative_to_original_model:
                        first_match_is_original_model_tp_or_tn = \
                            (original_model_logit > 0 and db_true_sentence_labels[0] == 1) or \
                            (original_model_logit <= 0 and db_true_sentence_labels[0] == 0)
                        is_relevant_distance = first_match_is_original_model_tp_or_tn
                        is_within_distance_threshold = \
                            (original_model_logit > 0 and db_distances[0] < options.class1_distance_threshold) or \
                            (original_model_logit <= 0 and db_distances[0] < options.class0_distance_threshold)

                    else:
                        # In using the nearest distance,
                        # we require the K-NN's prediction to match that of the sentence-level label of the first match
                        first_match_is_knn_tp_or_tn = (knn_logit > 0 and db_true_sentence_labels[0] == 1) or \
                                                      (knn_logit <= 0 and db_true_sentence_labels[0] == 0)
                        is_relevant_distance = first_match_is_knn_tp_or_tn
                        is_within_distance_threshold = \
                            (knn_logit > 0 and db_distances[0] < options.class1_distance_threshold) or \
                            (knn_logit <= 0 and db_distances[0] < options.class0_distance_threshold)
                    if is_relevant_distance and is_within_distance_threshold:
                        output_prediction_tuples.append(
                            (query_true_token_label, original_model_logit, np.abs(original_model_logit),
                             knn_logit, np.abs(knn_logit), db_distances[0]))
                    else:
                        admit_token = 0
            else:
                admit_token = 0

            # only add non-padding tokens to the total
            if query_non_padding_mask_for_token == 1:
                admit_constraints.append(admit_token)

        if (options.admitted_tokens_total_min < np.sum(admit_constraints) < options.admitted_tokens_total_max) and \
            np.mean(admit_constraints) > options.admitted_tokens_proportion_min:

            sentence_level_y_constrained.append(true_sentence_level_label)
            admitted_sentence_level_prediction = 1.0 if sentence_probs_for_sent[1]>sentence_probs_for_sent[0] else 0.0
            sentence_level_prediction_constrained.append(admitted_sentence_level_prediction)

            if admitted_sentence_level_prediction != true_sentence_level_label:
                print(f"Admitted but incorrect prediction sentence index: {sent_i}, "
                      f"true label: {true_sentence_level_label}")

        test_seq_y_for_eval.append(test_seq_y_for_sent_i)
        sentence_probs.append(sentence_probs_for_sent)
        all_contribution_tuples_by_sentence.append(contribution_tuples_for_sent)

    eval_stats = utils_eval.calculate_seq_metrics(test_seq_y_for_eval, all_contribution_tuples_by_sentence,
                                                  sentence_probs, tune_offset=False,
                                                  print_baselines=False,
                                                  output_generated_detection_file="",
                                                  numerically_stable=True, fce_eval=True)

    sentence_level_acc = calculate_accuracy(sentence_level_prediction, sentence_level_y)
    print(f"Sentence-level Accuracy: {sentence_level_acc} out of {len(sentence_level_prediction)} sentences")
    print(f"Proportion of ground-truth class 1: {np.mean(sentence_level_y)}")
    print(f"Proportion of predicted class 1: {np.mean(sentence_level_prediction)}")

    # constrained:
    print("-------Constrained sentences-------")
    sentence_level_acc_constrained = \
        calculate_accuracy(sentence_level_prediction_constrained, sentence_level_y_constrained)
    print(f"Sentence-level Accuracy (Among admitted): "
          f"{sentence_level_acc_constrained} out of {len(sentence_level_prediction_constrained)} sentences")
    print(f"Proportion of ground-truth class 1 among constrained: {np.mean(sentence_level_y_constrained)}")
    print(f"Proportion of predicted class 1 among constrained: {np.mean(sentence_level_prediction_constrained)}")


def calculate_accuracy(prediction_list, true_list):
    assert len(prediction_list) == len(true_list)
    return np.sum([1 if predicted == true_y else 0 for predicted, true_y in
                         zip(prediction_list, true_list)]) / len(true_list)


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
    parser.add_argument("--constrain_by_nearest_distance", default=False,
                        action='store_true',
                        help="If provided, only predictions for which the distance to the nearest match is less "
                             "than the provided value are considered. In the paper, we derive this from "
                             "correct *approximations* on the dev set. Importantly, this value is available to "
                             "all models, including the pure zero-shot setting, because it is determined by "
                             "sign flips to the original model, not the ground-truth data. In this way, we can "
                             "leverage the original model and the constructed K-NN as heuristics for "
                             "uncertainty, even without using ground-truth token-level data. Importantly, note that "
                             "in this version, we also require that the nearest match has the same "
                             "sentence-level label as the direction of the K-NN or original model output.")
    parser.add_argument("--class0_distance_threshold", default=0.0, type=float, help="class0_distance_threshold")
    parser.add_argument("--class1_distance_threshold", default=0.0, type=float, help="class1_distance_threshold")
    parser.add_argument("--nearest_distance_constraint_is_relative_to_original_model", default=False,
                        action='store_true',
                        help="If provided, the distance constraint is made relative to a match of the original "
                             "model. This is provided for reference, but we do not use this in the paper, and only "
                             "focus on the K-NN for the distance constraint as an example.")
    parser.add_argument("--constrain_by_knn_output_magnitude", default=False, action='store_true',
                        help="If provided, the K-NN output is constrained according to "
                             "--class0_magnitude_threshold and --class1_magnitude_threshold.")
    parser.add_argument("--constrain_by_original_model_output_magnitude", default=False, action='store_true',
                        help="If provided, the K-NN output is constrained according to "
                             "--class0_magnitude_threshold and --class1_magnitude_threshold.")
    parser.add_argument("--class0_magnitude_threshold", default=0.0, type=float, help="class0_magnitude_threshold")
    parser.add_argument("--class1_magnitude_threshold", default=0.0, type=float, help="class1_magnitude_threshold")
    parser.add_argument("--admitted_tokens_proportion_min", default=-1.0, type=float,
                        help="If provided, the proportion of admitted tokens must exceed this value. This is "
                             "expected to be a proportion in [0.0, 1.0].")
    parser.add_argument("--admitted_tokens_total_min", default=0.0, type=float,
                        help="If provided, the total number of admitted tokens must exceed this value. Default is 0.")
    parser.add_argument("--admitted_tokens_total_max", default=np.inf, type=float,
                        help="If provided, the total number of admitted tokens must be less than this value.")

    options = parser.parse_args()

    if options.admitted_tokens_proportion_min != -1:
        assert 0.0 <= options.admitted_tokens_proportion_min <= 1.0
    prediction_stats = \
                    utils_linear_exa.load_data_structure_torch_from_file(options.output_prediction_stats_file, -1)
    print(f"Loaded prediction stats file from {options.output_prediction_stats_file}")

    print(f"{''.join(['-']*45)}Analysis Type: {options.analysis_type}{''.join(['-']*45)}")
    score_output(prediction_stats, options)


if __name__ == "__main__":
    main()

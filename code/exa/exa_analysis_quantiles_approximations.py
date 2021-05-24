"""
Calculate quantiles to evaluate heuristics for uncertainty/prediction reliability of *approximations*.
This is similar in spirit to exa_analysis_quantiles.py, but note that here the analysis is against sign matches
between the original model and the K-NN approximation, and NOT the ground-truth.

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


def get_prediction_tuple_index_by_string(index_string):
    # Use this to get the index of elements of output_prediction_tuples
    if index_string == "query_true_token_label":
        return 0
    elif index_string == "original_model_logit":
        return 1
    elif index_string == "np.abs(original_model_logit)" or index_string == "original_model_logit_magnitude":
        return 2
    elif index_string == "knn_logit":
        return 3
    elif index_string == "np.abs(knn_logit)" or index_string == "knn_logit_magnitude":
        return 4
    elif index_string == "db_distances[0]" or index_string == "distance_of_nearest_match":
        return 5
    assert False, f"ERROR: Unknown index."


def score_ranked_predictions(output_prediction_tuples, sort_key=None, restrict_to_class=None,
                             tanh_of_prediction_logit=False, format_output_for_paper=False,
                             flip_class_for_metrics=False):
    # Note that tanh_of_prediction_logit should typically be False. It was just used here for debugging to get a
    # comparable sense of the saturated threshold cutoffs. Here, the
    # transform is applied after the sort, so it typically wouldn't alter the main metrics. (Before the sort is
    # possible, but note then that saturated values would be indistinguishable, which may not be desirable for
    # sorting.)
    assert sort_key in ["knn_logit_magnitude", "distance_of_nearest_match_using_knn"]
    # do not change the input
    output_prediction_tuples = copy.deepcopy(output_prediction_tuples)

    if sort_key == "knn_logit_magnitude":
        # Note the negative in the comparison
        comparator_sign = -1
        prediction_key = "knn_logit"
        logit_magnitude_key = sort_key
        print(f"Using KNN approximation accuracy when considering the KNN quantiles.")
        if restrict_to_class is not None:
            assert restrict_to_class in [0,1]
            if restrict_to_class == 0:
                constrained_output_prediction_tuples = []
                for prediction_tuple in output_prediction_tuples:
                    if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] <= 0:
                        constrained_output_prediction_tuples.append(prediction_tuple)
                output_prediction_tuples = constrained_output_prediction_tuples
                print(f"***Only considering KNN logit <= 0***")
            elif restrict_to_class == 1:
                constrained_output_prediction_tuples = []
                for prediction_tuple in output_prediction_tuples:
                    if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] > 0:
                        constrained_output_prediction_tuples.append(prediction_tuple)
                output_prediction_tuples = constrained_output_prediction_tuples
                print(f"***Only considering KNN logit > 0***")

    elif sort_key == "distance_of_nearest_match_using_knn":
        sort_key = "distance_of_nearest_match"
        # Note the positive in the comparison
        comparator_sign = 1
        prediction_key = "knn_logit"
        logit_magnitude_key = "knn_logit_magnitude"
        print(f"Using KNN approximation accuracy when considering the "
              f"distance-to-nearest-match quantiles.")
        if restrict_to_class is not None:
            assert restrict_to_class in [0,1]
            if restrict_to_class == 0:
                constrained_output_prediction_tuples = []
                for prediction_tuple in output_prediction_tuples:
                    if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] <= 0:
                        constrained_output_prediction_tuples.append(prediction_tuple)
                output_prediction_tuples = constrained_output_prediction_tuples
                print(f"***Only considering KNN logit <= 0***")
            elif restrict_to_class == 1:
                constrained_output_prediction_tuples = []
                for prediction_tuple in output_prediction_tuples:
                    if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] > 0:
                        constrained_output_prediction_tuples.append(prediction_tuple)
                output_prediction_tuples = constrained_output_prediction_tuples
                print(f"***Only considering KNN logit > 0***")
    else:
        assert False

    comparison_index = get_prediction_tuple_index_by_string(sort_key)

    # We sort from smallest to largest. For output magnitudes, note that we multiply by -1. The thresholds from
    # left-to-right are then from the greatest magnitude to the lowest. For the output tables, we multiply
    # by -1 again.
    sorted_predictions_tuples = sorted(output_prediction_tuples,
                                       key=lambda x: (comparator_sign * x[comparison_index]), reverse=False)

    if tanh_of_prediction_logit:
        sorted_logit_magnitudes = [comparator_sign *
                                   torch.tanh(torch.tensor(
                                       [x[get_prediction_tuple_index_by_string(logit_magnitude_key)]])
                                   ).item()
                                   for x in sorted_predictions_tuples]
    else:
        sorted_logit_magnitudes = [comparator_sign * x[get_prediction_tuple_index_by_string(logit_magnitude_key)]
                                   for x in sorted_predictions_tuples]
    sorted_distances_of_nearest_match = [x[get_prediction_tuple_index_by_string("distance_of_nearest_match")]
                                         for x in sorted_predictions_tuples]
    sorted_model_predictions = []
    sorted_true_labels = []  # here, 'true' labels are the original model's predictions
    sorted_model_acc = []

    for prediction_tuple in sorted_predictions_tuples:
        if not flip_class_for_metrics:
            prediction = 1 if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] > 0 else 0
            # true_label = prediction_tuple[get_prediction_tuple_index_by_string("query_true_token_label")]
            # Importantly, note that the 'true' label here is the original model's predictions, not the ground-truth
            # label, which is commented above.
            true_label = 1 if prediction_tuple[get_prediction_tuple_index_by_string("original_model_logit")] > 0 else 0
        else:
            # this is for getting a reference of the F score for the majority class in the case of the grammar sets
            prediction = 0 if prediction_tuple[get_prediction_tuple_index_by_string(prediction_key)] > 0 else 1
            true_label = 0 if prediction_tuple[get_prediction_tuple_index_by_string("original_model_logit")] > 0 else 1
        sorted_model_predictions.append(prediction)
        sorted_true_labels.append(true_label)
        sorted_model_acc.append(int(prediction == true_label))

    sorted_quantiles = {}
    # by 'running' we mean 'cumulative'
    sorted_quantiles["distance_of_nearest_match"] = []
    sorted_quantiles["running_distance_of_nearest_match"] = []
    sorted_quantiles["average_in_quantile_distance_of_nearest_match"] = []
    sorted_quantiles["fscore_0_5"] = []
    sorted_quantiles["acc"] = []
    sorted_quantiles["n"] = []
    sorted_quantiles["running_true_positive_class_count"] = []
    sorted_quantiles["running_predicted_positive_class_count"] = []
    sorted_quantiles[f"{logit_magnitude_key}"] = []
    sorted_quantiles[f"running_{logit_magnitude_key}"] = []
    sorted_quantiles[f"average_in_quantile_running_{logit_magnitude_key}"] = []

    quantile_size = len(sorted_predictions_tuples) // 4  # floor
    quantile_indexes = [quantile_size, quantile_size*2, quantile_size*3, len(sorted_predictions_tuples)]

    for q, instance_i in enumerate(quantile_indexes):
        eval_stats = utils_eval.calculate_metrics(sorted_true_labels[0:instance_i],
                                                  sorted_model_predictions[0:instance_i], True,
                                                  print_results=False)
        sorted_quantiles["fscore_0_5"].append(eval_stats["fscore_0_5"])
        sorted_quantiles["n"].append(instance_i)
        sorted_quantiles["distance_of_nearest_match"].append(sorted_distances_of_nearest_match[instance_i-1])
        sorted_quantiles["running_distance_of_nearest_match"].append(
            np.mean(sorted_distances_of_nearest_match[0:instance_i])
        )
        if q == 0:
            prev_quantile = 0
        else:
            prev_quantile = quantile_indexes[q-1]
        sorted_quantiles["average_in_quantile_distance_of_nearest_match"].append(
            np.mean(sorted_distances_of_nearest_match[prev_quantile:instance_i])
        )
        sorted_quantiles[f"{logit_magnitude_key}"].append(sorted_logit_magnitudes[instance_i-1])
        sorted_quantiles[f"running_{logit_magnitude_key}"].append(
            np.mean(sorted_logit_magnitudes[0:instance_i])
        )
        sorted_quantiles[f"average_in_quantile_running_{logit_magnitude_key}"].append(
            np.mean(sorted_logit_magnitudes[prev_quantile:instance_i])
        )
        sorted_quantiles["acc"].append(
            np.mean(sorted_model_acc[0:instance_i])
        )
        sorted_quantiles["running_true_positive_class_count"].append(
            np.sum(sorted_true_labels[0:instance_i])
        )
        sorted_quantiles["running_predicted_positive_class_count"].append(
            np.sum(sorted_model_predictions[0:instance_i])
        )

    assert sorted_quantiles["n"][-1] == len(sorted_predictions_tuples)
    if not format_output_for_paper:
        for key in sorted_quantiles:
            print(f"{key}")
            print(f"\t{sorted_quantiles[key]}")
    else:
        print(f"F_0.5: {', '.join([str(round(x, 6)) for x in sorted_quantiles['fscore_0_5']])}")
        print(f"Accuracy: {', '.join([str(round(x, 6)) for x in sorted_quantiles['acc']])}")
        if sort_key == "distance_of_nearest_match":
            print(f"Distance Thresholds: "
                  f"{', '.join([str(round(x, 6)) for x in sorted_quantiles['distance_of_nearest_match']])}")
        else:
            # note the multiplication
            print(f"Thresholds: {', '.join([str(round(-1*x, 6)) for x in sorted_quantiles[logit_magnitude_key]])}")
        print(f"N: {', '.join([str(x) for x in sorted_quantiles['n']])}")


def score_output(prediction_stats, options):
    print(f"Number of sentences under consideration for analysis: {len(prediction_stats)}")
    test_seq_y_for_eval = []
    sentence_probs = []
    all_contribution_tuples_by_sentence = []

    output_prediction_tuples = []
    num_max_length_padding_tokens_ignored = 0
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
                if options.constrain_by_nearest_distance == -1:
                    output_prediction_tuples.append(
                        (query_true_token_label, original_model_logit, np.abs(original_model_logit),
                         knn_logit, np.abs(knn_logit), db_distances[0]))
                else:
                    # In using the nearest distance,
                    # we require the K-NN's prediction to match that of the sentence-level label of the first match
                    first_match_is_knn_tp_or_tn = (knn_logit > 0 and db_true_sentence_labels[0] == 1) or (
                            knn_logit <= 0 and db_true_sentence_labels[0] == 0)
                    if first_match_is_knn_tp_or_tn and db_distances[0] < options.constrain_by_nearest_distance:
                        output_prediction_tuples.append(
                            (query_true_token_label, original_model_logit, np.abs(original_model_logit),
                             knn_logit, np.abs(knn_logit), db_distances[0]))

        test_seq_y_for_eval.append(test_seq_y_for_sent_i)
        sentence_probs.append(sentence_probs_for_sent)
        all_contribution_tuples_by_sentence.append(contribution_tuples_for_sent)

    eval_stats = utils_eval.calculate_seq_metrics(test_seq_y_for_eval, all_contribution_tuples_by_sentence,
                                                  sentence_probs, tune_offset=False,
                                                  print_baselines=False,
                                                  output_generated_detection_file="",
                                                  numerically_stable=True, fce_eval=True)

    if not options.do_not_exclude_padding:
        print(f"Number of max length padding tokens ignored in the following analyses: "
              f"{num_max_length_padding_tokens_ignored}")

    return output_prediction_tuples


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
    parser.add_argument("--calculate_raw_distance_quantiles", default=False, action='store_true',
                        help="We can also calculate quantiles using the raw distances. However, note that this is "
                             "only for k=0.")
    parser.add_argument("--show_combined_magnitude_direction_quantiles", default=False, action='store_true',
                        help="In this case, we produce the metrics when considering both positive and "
                             "negative magnitudes together. Note that the 4th quantile contains the total Acc/F0.5 "
                             "across all of the data.")
    parser.add_argument("--format_output_for_paper", default=False, action='store_true',
                        help="This primarily only prints the subset of the core output used in the paper.")
    parser.add_argument("--constrain_by_nearest_distance", default=-1, type=float,
                        help="If provided, only predictions for which the distance to the nearest match is less "
                             "than the provided value are considered. In the paper, we derive this from "
                             "correct *approximations* on the dev set. Importantly, this value is available to "
                             "all models, including the pure zero-shot setting, because it is determined by "
                             "sign flips to the original model, not the ground-truth data. In this way, we can "
                             "leverage the original model and the constructed K-NN as heuristics for "
                             "uncertainty, even without using ground-truth token-level data. Importantly, note that "
                             "in this version, we also require that the nearest match has the same "
                             "sentence-level label as the direction of the K-NN's output.")
    parser.add_argument("--flip_class_for_metrics", default=False, action='store_true',
                        help="Use with caution. This treats the original class 0 as the 'positive' class when "
                             "calculating F scores.")
    parser.add_argument("--do_not_exclude_padding", default=False, action='store_true',
                        help="Typically this should NOT be provided, as we want to analyze the approximations "
                             "only in terms of tokens that the original model actually saw (i.e., we want "
                             "to exclude padding).")

    options = parser.parse_args()
    if options.constrain_by_nearest_distance != -1:
        assert options.constrain_by_nearest_distance > 0, "ERROR: Distance constraint must be > 0."
        print(f"Constraining all predictions to those for which the L^2 distance to the nearest match is less than "
              f"{options.constrain_by_nearest_distance} AND the sentence-level label of the nearest match is the "
              f"same as that of the K-NN.")

    if options.flip_class_for_metrics:
        print(f"WARNING: class 0 is being used as the 'positive' class when calculating F scores.")
    prediction_stats = \
                    utils_linear_exa.load_data_structure_torch_from_file(options.output_prediction_stats_file, -1)
    print(f"Loaded prediction stats file from {options.output_prediction_stats_file}")

    print(f"{''.join(['-']*45)}Analysis Type: {options.analysis_type}{''.join(['-']*45)}")
    output_prediction_tuples = score_output(prediction_stats, options)

    transform_logit = False
    if options.show_combined_magnitude_direction_quantiles:
        print(f"{''.join(['+'] * 45)}Transform via tanh={transform_logit}{''.join(['+'] * 45)}")
        print(f"{''.join(['-'] * 45)}Analysis Type: knn_logit_magnitude; "
              f"tanh={transform_logit} {''.join(['-'] * 45)}")
        score_ranked_predictions(output_prediction_tuples, sort_key="knn_logit_magnitude",
                                 tanh_of_prediction_logit=transform_logit,
                                 format_output_for_paper=options.format_output_for_paper,
                                 flip_class_for_metrics=options.flip_class_for_metrics)
        if options.calculate_raw_distance_quantiles:
            print(f"{''.join(['-'] * 45)}Analysis Type: distance_of_nearest_match_using_knn; "
                  f"tanh={transform_logit}{''.join(['-'] * 45)}")
            score_ranked_predictions(output_prediction_tuples, sort_key="distance_of_nearest_match_using_knn",
                                     tanh_of_prediction_logit=transform_logit,
                                     format_output_for_paper=options.format_output_for_paper,
                                     flip_class_for_metrics=options.flip_class_for_metrics)

    for class_to_restrict in [0, 1]:
        print(f"{''.join(['-'] * 45)}Analysis Type: knn_logit_magnitude; "
              f"tanh={transform_logit} "
              f"** restrict to class {class_to_restrict} ** {''.join(['-'] * 45)}")
        score_ranked_predictions(output_prediction_tuples, sort_key="knn_logit_magnitude",
                                 restrict_to_class=class_to_restrict, tanh_of_prediction_logit=transform_logit,
                                 format_output_for_paper=options.format_output_for_paper,
                                 flip_class_for_metrics=options.flip_class_for_metrics)
        if options.calculate_raw_distance_quantiles:
            print(f"{''.join(['-'] * 45)}Analysis Type: distance_of_nearest_match_using_knn; "
                  f"tanh={transform_logit} "
                  f"** restrict to class {class_to_restrict} ** {''.join(['-'] * 45)}")
            score_ranked_predictions(output_prediction_tuples, sort_key="distance_of_nearest_match_using_knn",
                                     restrict_to_class=class_to_restrict,
                                     tanh_of_prediction_logit=transform_logit,
                                     format_output_for_paper=options.format_output_for_paper,
                                     flip_class_for_metrics=options.flip_class_for_metrics)

    # Note that here we're somewhat loose with the term 'logit', which in the use here just means the
    # untransformed output of the original network in [-inf, inf], or the bounded value of the K-NNs in
    # [-2+constant, 2+constant].
    print(f"All results here are with a decision boundary of 0")
    if options.flip_class_for_metrics:
        print(f"WARNING: class 0 was used as the 'positive' class when calculating F scores.")
    if options.do_not_exclude_padding:
        print(f"WARNING: Padding tokens are included in the metrics. These tokens were never seen by the original "
              f"model.")


if __name__ == "__main__":
    main()

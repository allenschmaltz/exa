# Evaluate the K-NN as a sequence labeler. This also has options for saving an annotated version of the input (with
# ground-truth labels and predictions from the original model and the K-NN), as well as saving the matched exemplars
# to a human readable file with the terms of the K-NN logit decomposed across exemplars along with other associated
# meta data such as the ground-truth labels from the database/support set (i.e., all the output an end-user would need
# to see to use this method for interpretability/explainability).
# Finally, there is an option to save the output predictions and top-K exemplars distances to
# disk for further analysis off the GPU's.

from model_exa import LinearExA
import utils_exemplar
import utils_linear_exa
import utils_eval
import constants
import constants_exa

import numpy as np
import codecs
from collections import defaultdict
import torch
import torch.nn as nn

import subprocess
import time

from collections import namedtuple


def init_eval_data_structures(data, options, number_of_filter_maps):
    print(f"Loading exemplar eval data.")
    query_sentence_to_stats, query_sentence_to_tokens, query_token_to_sentence, query_token_to_sent_relative_token_id, \
        query_token_to_stats, _, _, _ = \
        utils_linear_exa.get_exemplar_data_only_stats_from_file(options.exemplar_data_query_file,
                                                                number_of_filter_maps[0])
    assert len(query_sentence_to_stats) == len(data["test_sentences"])
    print(f"Total number of query tokens: {len(query_token_to_stats)}")
    print(f"Exemplar data loaded.")
    query_non_padding_masks_by_sentence = utils_linear_exa.get_real_token_non_padding(data, options.max_length)
    return query_sentence_to_stats, query_sentence_to_tokens, query_token_to_sentence, \
           query_token_to_sent_relative_token_id, query_token_to_stats, query_non_padding_masks_by_sentence


def init_database_data_structures(data, options, number_of_filter_maps):
    print(f"Loading exemplar database data for analysis.")
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, _, _, _ = \
        utils_linear_exa.get_exemplar_data_only_stats_from_file(options.exemplar_data_database_file,
                                                                number_of_filter_maps[0])

    total_database_tokens = len(database_token_to_stats)
    print(f"Total number of database tokens: {total_database_tokens}")
    print(f"Exemplar database data loaded.")
    print(f"Loading original database sentences and labels.")
    database_original_sentences, database_sentence_labels = utils_exemplar.get_original_sentences(
        options.exemplar_sentences_database_file)
    print(f"Original database sentences and labels loaded.")
    return database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
           database_token_to_sent_relative_token_id, database_token_to_stats, database_original_sentences, \
           database_sentence_labels


def eval_linear_exa(eval_data_structures, model, params, options, data, number_of_filter_maps,
                    output_generated_detection_file, database_data_structures=None,
                    eval_split_chunk_ids_subset=None, tune_offset=True, print_memory_stats=True):
    # Note that if eval_split_chunk_ids_subset is not None, the full query file is not evaluated and only the
    # chunk_ids in eval_split_chunk_ids_subset are considered.
    start_eval_time = time.time()

    number_of_padding_tokens_excluded_from_analysis = 0
    # This should always be True in the current version. To keep things simple, if token-level labels are not
    # available (as in the zero-shot case), just give the script dummy labels of 0 or 1 for each input
    # token and ignore the token-level eval output.
    eval_token_level_flips = True
    print(f"Evaluating token level flips")
    model.eval()

    if options.save_prediction_stats:
        assert options.print_error_analysis and eval_token_level_flips
        assert options.output_prediction_stats_file != ""
        assert options.max_exemplars_to_return == model.top_k, f"ERROR: The full K is expected when saving stats."
        # We save the output and perform the various ExA analyses on the archived output, off the GPUs.
        prediction_stats = []

    if print_memory_stats:
        print(f'Initial allocation (note BERT is currently loaded, which is typically not needed for LinearExA, '
              f'since the distances are pre-calculated.) '
              f'torch.cuda.max_memory_allocated: '
              f'{torch.cuda.max_memory_allocated(params["main_device"])}')
    # TODO: BERT can be dropped after verifying no changes to the data loading -- we don't need the tokenizer but
    # check that 'bert_model is None' doesn't create issues with the pre-processing/data loading scripts.

    y = data["test_y"]
    if eval_split_chunk_ids_subset is None:
        test_seq_y_for_eval = data["test_seq_y"]
    else:
        test_seq_y_for_eval = []  # this is built up below for each sent_i under consideration
    test_seq_y_for_eval_no_padding = []  # for reference, exclude tokens never seen by original model

    query_sentence_to_stats, query_sentence_to_tokens, query_token_to_sentence, \
        query_token_to_sent_relative_token_id, query_token_to_stats, query_non_padding_masks_by_sentence = \
        eval_data_structures

    if eval_split_chunk_ids_subset is None:
        chunk_ids = utils_linear_exa.load_memory_structure_torch(options.distance_dir,
                                                                f"distances_to_db_chunk_ids", 0, -1).numpy()
    else:
        print(f"Evaluating a restricted set of chunk_ids.")
        chunk_ids = eval_split_chunk_ids_subset
    sentence_probs = []
    token_logits_by_chunk = []
    token_nearest_distances_by_chunk = []
    token_nearest_indexes_by_chunk = []
    # loop through query sentences
    for chunk_id in chunk_ids:
        if chunk_id % 10 == 0:
            print(f"TopK: Processing distance chunk id {chunk_id} of {len(chunk_ids)}.")
        distances_by_sentences_chunk = \
            utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db",
                                                         chunk_id, int(params["GPU"]))
        support_set_indexes_by_sentences_chunk = \
            utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db_support_indexes",
                                                         chunk_id, int(params["GPU"]))

        if chunk_id == 0 and print_memory_stats:
            print(f'distance structures for chunk 0 torch.cuda.max_memory_allocated: '
                  f'{torch.cuda.max_memory_allocated(params["main_device"])}')

        with torch.no_grad():
            if options.print_error_analysis:
                predicted_token_contributions, nearest_distances, nearest_indexes = \
                    model.forward(top_k_distances=distances_by_sentences_chunk,
                                  top_k_distances_idx=support_set_indexes_by_sentences_chunk,
                                  return_nearest_distances=True,
                                  return_nearest_indexes=True,
                                  max_exemplars_to_return=options.max_exemplars_to_return)

                nearest_indexes = nearest_indexes.cpu()
            else:
                predicted_token_contributions, nearest_distances = \
                    model.forward(top_k_distances=distances_by_sentences_chunk,
                                  top_k_distances_idx=support_set_indexes_by_sentences_chunk,
                                  return_nearest_distances=True,
                                  max_exemplars_to_return=1)
            predicted_token_contributions = predicted_token_contributions.cpu()
            nearest_distances = nearest_distances.cpu()
        token_logits_by_chunk.append(predicted_token_contributions)
        token_nearest_distances_by_chunk.append(nearest_distances)
        if options.print_error_analysis:
            token_nearest_indexes_by_chunk.append(nearest_indexes)
        if chunk_id == 0 and print_memory_stats:
            nvidia_smi_out = subprocess.run(["nvidia-smi"], capture_output=True)
            print(f'{nvidia_smi_out.stdout.decode("utf-8")}')

    # the logits are batched by chunk, so we split by sentence for eval
    all_contribution_tuples_by_sentence = []
    sign_flips_by_sentence = []
    if eval_token_level_flips:
        true_sign_flips_by_sentence = []
    sentence_level_prediction = []
    original_model_sentence_level_prediction_from_token_level = []  # for reference; using token-level contributions
    true_sentence_labels = []
    nearest_distance_correct_model_sign = []
    nearest_distance_wrong_model_sign = []
    nearest_distance_correct_true_sign = []
    nearest_distance_wrong_true_sign = []
    # among distances relative to the original model, record actually correct predictions (based on ground-truth);
    # i.e., the original model is right/wrong, and/but the new knn is correct
    nearest_distance_correct_model_sign_and_true_sign = []
    nearest_distance_wrong_model_sign_and_true_sign = []
    # Note: For the token-level accuracy of the *original* model (i.e., without regard to the KNN), the following
    # must be constant across training epochs, else there is an error in the analysis code:
    total_sign_flips_of_the_original_model = 0

    # The following are the predictions in {0,1} of the original model excluding padding. This is only used to get
    # random and majority class baselines for model approximations. E.g., if we were to say the approximation always
    # produces class 0, how well would it do in terms of accuracy? This is particularly useful for context for
    # unbalanced datasets.
    original_model_non_padding_predictions = []

    if options.print_error_analysis:
        output_exemplar_analysis_file_object = codecs.open(options.output_analysis_file, 'w', 'utf-8')
        if options.output_save_type == 3:
            rng = np.random.default_rng(seed=options.seed_value)
    if options.save_annotations:
        output_annotations_file_object = codecs.open(options.output_annotations_file, 'w', 'utf-8')
    assert len(token_logits_by_chunk) == len(chunk_ids)
    # when processing the chunks, we index the forward output lists via chunk_process_index to handle the case
    # when chunk_ids are not sequential, but chunk_id should always be used to access the saved structure
    for chunk_process_index, chunk_id in enumerate(chunk_ids):
        sentence_ids_in_current_chunk = utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db_sentence_ids",
                                                                         chunk_id, -1).numpy()
        token_logits = token_logits_by_chunk[chunk_process_index]  #token_logits_by_chunk[chunk_id]
        token_nearest_distances = token_nearest_distances_by_chunk[chunk_process_index]  # batch/chunk by K
        if options.print_error_analysis:
            token_nearest_indexes = token_nearest_indexes_by_chunk[chunk_process_index]  # batch/chunk by K
        running_sentence_ids_index = 0
        for sent_i in sentence_ids_in_current_chunk:
            if eval_split_chunk_ids_subset is not None:
                test_seq_y_for_eval.append(data["test_seq_y"][sent_i])
            contribution_tuples_for_sent = []
            sentence_probs.append([query_sentence_to_stats[sent_i][1], query_sentence_to_stats[sent_i][2]])
            sent_tokens = data["test_sentences"][sent_i]  # this works with shuffle/subset indexes, since sent_i indexed
            query_non_padding_masks = query_non_padding_masks_by_sentence[sent_i]
            assert len(sent_tokens) == len(query_sentence_to_tokens[sent_i])
            assert query_sentence_to_stats[sent_i][0] == y[sent_i]
            assert len(sent_tokens) == len(query_non_padding_masks)
            token_ids = query_sentence_to_tokens[sent_i]

            signs = []
            for token_logit in token_logits[running_sentence_ids_index:running_sentence_ids_index+len(token_ids)]:
                contribution_tuples_for_sent.append((0.0, token_logit.item(), 0.0, 0.0))
                signs.append(token_logit.gt(0).float().item())
            distances = []
            for token_distance in token_nearest_distances[running_sentence_ids_index:running_sentence_ids_index + len(token_ids)]:
                # Note that token_distance is the full row of K exemplar distances up to max_exemplars_to_return
                # I.e., a vector of size min(K, max_exemplars_to_return)
                distances.append(token_distance)
            if options.print_error_analysis:
                nearest_database_indexes = []
                for token_db_index in token_nearest_indexes[running_sentence_ids_index:running_sentence_ids_index + len(token_ids)]:
                    # As with token_distance, token_db_index is a vector of size min(K, max_exemplars_to_return)
                    nearest_database_indexes.append(token_db_index)
            running_sentence_ids_index += len(token_ids)

            all_contribution_tuples_by_sentence.append(contribution_tuples_for_sent)

            true_sentence_level_label = query_sentence_to_stats[sent_i][0]
            true_sentence_labels.append(float(true_sentence_level_label))
            if np.sum(signs) > 0:
                sentence_level_prediction.append(1.0)
            else:
                sentence_level_prediction.append(0.0)

            if options.save_annotations:  # annotate and save sentence
                output_annotations_file_object = annotate_sentence_with_token_labels(
                    output_annotations_file_object=output_annotations_file_object,
                    sent_tokens=list(sent_tokens),
                    knn_prediction_signs=signs, token_ids=token_ids, query_token_to_stats=query_token_to_stats,
                    query_non_padding_masks=query_non_padding_masks)
            if options.save_prediction_stats:
                # "test_seq_y_for_sent_i" is actually extraneous since the true token-level labels are included
                # in "token_level_stats", but I include it as a check
                prediction_stats_for_sent = {"sent_i": sent_i,
                                             "true_sentence_level_label": true_sentence_level_label,
                                             "test_seq_y_for_sent_i": data["test_seq_y"][sent_i],
                                             "sentence_probs_for_sent_i": [query_sentence_to_stats[sent_i][1],
                                                                           query_sentence_to_stats[sent_i][2]],
                                             "knn_bias": model.model_bias.item(),
                                             "token_level_stats": []
                                             }

            # record sign flips
            # Note: When recording sign flips, we exclude tokens that exceeded the max length of the original model,
            # since the model never makes explicit predictions over those tokens. (These cases are rare in the
            # current models/datasets. In practice here, including them would slightly inflate the real
            # metric, so it's better exclude them since they don't reflect the approximation's match to the original
            # model.)
            num_flips = 0
            if eval_token_level_flips:
                num_true_flips = 0
            original_model_detection = False
            for token_i, token_id in enumerate(token_ids):
                analysis_labels = []
                if eval_token_level_flips:
                    query_true_token_label, query_token_contribution = query_token_to_stats[token_id]
                    if query_true_token_label == 1:  # ground truth
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            test_seq_y_for_eval_no_padding.append(1)
                        if signs[token_i] == 0:
                            if query_non_padding_masks[token_i] != 0:  # exclude padding
                                num_true_flips += 1
                                nearest_distance_wrong_true_sign.append(distances[token_i][0])  # use nearest (k=0)
                            analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_TRUTH)
                        else:
                            if query_non_padding_masks[token_i] != 0:  # exclude padding
                                nearest_distance_correct_true_sign.append(distances[token_i][0])
                            analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_MATCHES_TRUTH)
                        if query_token_contribution <= 0 and query_non_padding_masks[token_i] != 0:  # model prediction, excluding padding
                            total_sign_flips_of_the_original_model += 1
                    else:
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            test_seq_y_for_eval_no_padding.append(0)
                        if signs[token_i] == 1:
                            if query_non_padding_masks[token_i] != 0:  # exclude padding
                                num_true_flips += 1
                                nearest_distance_wrong_true_sign.append(distances[token_i][0])
                            analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_TRUTH)
                        else:
                            if query_non_padding_masks[token_i] != 0:  # exclude padding
                                nearest_distance_correct_true_sign.append(distances[token_i][0])
                            analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_MATCHES_TRUTH)
                        if query_token_contribution > 0 and query_non_padding_masks[token_i] != 0:  # model prediction, excluding padding
                            total_sign_flips_of_the_original_model += 1
                else:
                    _, query_token_contribution = query_token_to_stats[token_id]
                if query_non_padding_masks[token_i] == 0:
                    number_of_padding_tokens_excluded_from_analysis += 1
                else:  # exclude padding
                    # this is used to get random and majority class baselines for the approximation
                    original_model_non_padding_predictions.append(1.0 if query_token_contribution > 0 else 0.0)
                if query_token_contribution > 0:  # model prediction
                    original_model_detection = True
                    if signs[token_i] == 0:
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            num_flips += 1
                            nearest_distance_wrong_model_sign.append(distances[token_i][0])
                        analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_ORIGINAL_MODEL)

                        if eval_token_level_flips:
                            if query_true_token_label == 0:  # ground truth
                                if query_non_padding_masks[token_i] != 0:  # exclude padding
                                    nearest_distance_wrong_model_sign_and_true_sign.append(distances[token_i][0])
                    else:
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            nearest_distance_correct_model_sign.append(distances[token_i][0])
                        analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_MATCHES_ORIGINAL_MODEL)
                        if eval_token_level_flips:
                            if query_true_token_label == 1:  # ground truth
                                if query_non_padding_masks[token_i] != 0:  # exclude padding
                                    nearest_distance_correct_model_sign_and_true_sign.append(distances[token_i][0])
                else:
                    if signs[token_i] == 1:
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            num_flips += 1
                            nearest_distance_wrong_model_sign.append(distances[token_i][0])
                        analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_ORIGINAL_MODEL)

                        if eval_token_level_flips:
                            if query_true_token_label == 1:  # ground truth
                                if query_non_padding_masks[token_i] != 0:  # exclude padding
                                    nearest_distance_wrong_model_sign_and_true_sign.append(distances[token_i][0])
                    else:
                        if query_non_padding_masks[token_i] != 0:  # exclude padding
                            nearest_distance_correct_model_sign.append(distances[token_i][0])
                        analysis_labels.append(constants_exa.MODEL_ANALYSIS_SIGN_MATCHES_ORIGINAL_MODEL)
                        if eval_token_level_flips:
                            if query_true_token_label == 0:  # ground truth
                                if query_non_padding_masks[token_i] != 0:  # exclude padding
                                    nearest_distance_correct_model_sign_and_true_sign.append(distances[token_i][0])

                if options.print_error_analysis:
                    assert len(analysis_labels) == len(set(analysis_labels)), f"ERROR: Duplicate analysis labels are " \
                                                                              f"not expected."
                    save_this_token = False
                    if options.output_save_type == 0:
                        save_this_token = True
                    elif options.output_save_type == 1:
                        if constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_ORIGINAL_MODEL in analysis_labels:
                            save_this_token = True
                    elif options.output_save_type == 2:
                        if constants_exa.MODEL_ANALYSIS_SIGN_FLIP_TO_TRUTH in analysis_labels:
                            save_this_token = True
                    elif options.output_save_type == 3:
                        if 1 == rng.binomial(1, options.binomial_sample_p, size=1):
                            save_this_token = True
                    # TODO: The remaining options (4 to 8) were for debugging/dev and can be removed in future versions.
                    elif options.output_save_type == 4:
                        if distances[token_i][0] < 1.0:
                            save_this_token = True
                    elif options.output_save_type == 5:
                        if distances[token_i][0] < 1.0 and token_i < options.max_length:
                            save_this_token = True
                    elif options.output_save_type == 6:
                        if distances[token_i][0] < 1.0 and query_token_contribution != 0:
                            save_this_token = True
                    elif options.output_save_type == 7:
                        if query_token_contribution == 0:  # This should only occur for padding tokens.
                            save_this_token = True
                    elif options.output_save_type == 8:
                        if query_non_padding_masks[token_i] == 0:
                            save_this_token = True
                    else:
                        assert False, f"ERROR: Invalid --output_save_type"
                    if save_this_token:
                        output_exemplar_analysis_file_object = save_error_analysis_summary_topk(
                                                                       "; ".join(analysis_labels),
                                                                       model, params, list(sent_tokens),
                                                                       database_data_structures,
                                                                       nearest_database_indexes[token_i],
                                                                       query_true_token_label,
                                                                       query_token_contribution, token_i,
                                                                       token_id, sent_i,
                                                                       contribution_tuples_for_sent,
                                                                       distances[token_i],
                                                                       output_exemplar_analysis_file_object,
                                                                       query_non_padding_masks[token_i])
                # collect prediction stats for archive, including K-NN term decomposition
                if options.save_prediction_stats:
                    token_level_stats = get_prediction_summary_topk("; ".join(analysis_labels),
                                                                    model, params, list(sent_tokens),
                                                                    database_data_structures,
                                                                    nearest_database_indexes[token_i],
                                                                    query_true_token_label,
                                                                    query_token_contribution, token_i,
                                                                    token_id, sent_i,
                                                                    contribution_tuples_for_sent,
                                                                    distances[token_i],
                                                                    query_non_padding_masks[token_i])
                    prediction_stats_for_sent["token_level_stats"].append(token_level_stats)

            if options.save_prediction_stats:
                prediction_stats.append(prediction_stats_for_sent)

            sign_flips_by_sentence.append(num_flips)
            if eval_token_level_flips:
                true_sign_flips_by_sentence.append(num_true_flips)
            if original_model_detection:
                original_model_sentence_level_prediction_from_token_level.append(1.0)
            else:
                original_model_sentence_level_prediction_from_token_level.append(0.0)

    if options.print_error_analysis:
        output_exemplar_analysis_file_object.close()

    if options.save_annotations:
        output_annotations_file_object.close()

    if options.save_prediction_stats:
        assert len(prediction_stats) != 0
        print(f"Number of sentences in the prediction stats: {len(prediction_stats)}")
        utils_linear_exa.save_data_structure_torch_to_file(options.output_prediction_stats_file, prediction_stats)
        print(f"Saved prediction stats to {options.output_prediction_stats_file}")

    if eval_split_chunk_ids_subset is None:
        assert len(query_token_to_stats) == np.sum([len(x) for x in test_seq_y_for_eval]), \
            f"ERROR: The number of tokens evaluated " \
            f"does not match the expected number " \
            f"in this split."
        total_tokens_considered_in_split = len(query_token_to_stats)
    else:
        total_tokens_considered_in_split = np.sum([len(x) for x in test_seq_y_for_eval])

    print(f"{''.join(['-'] * 35)}")
    print(f"Total tokens considered in full evaluation: {total_tokens_considered_in_split}")
    total_tokens_considered_in_split_excluding_padding = \
        total_tokens_considered_in_split-number_of_padding_tokens_excluded_from_analysis
    print(f"Tokens that exceeded the original model's max length: {number_of_padding_tokens_excluded_from_analysis}. "
          f"A prediction is made for the comparison metrics, but we exclude these tokens from the sign flip analysis "
          f"since the original model never sees those tokens, for a total of "
          f"{total_tokens_considered_in_split_excluding_padding} under consideration.")
    assert total_tokens_considered_in_split == len(nearest_distance_correct_model_sign) + \
           len(nearest_distance_wrong_model_sign) + number_of_padding_tokens_excluded_from_analysis
    approx_acc = np.sum([1 if predicted == true_y else 0 for predicted, true_y in
                         zip(sentence_level_prediction, true_sentence_labels)]) / len(true_sentence_labels)
    model_acc = np.sum([1 if predicted == true_y else 0 for predicted, true_y in
                         zip(original_model_sentence_level_prediction_from_token_level, true_sentence_labels)]) / len(true_sentence_labels)
    print(f"Sentence-level accuracy (based on token-level predictions): "
          f"{approx_acc} (out of {len(true_sentence_labels)})")
    print(f"Reference sentence-level accuracy (based on token-level predictions) of the original model: "
          f"{model_acc} (out of {len(true_sentence_labels)})")

    print(f"Total sign flips (relative to original model): {np.sum(sign_flips_by_sentence)} out of "
          f"{total_tokens_considered_in_split_excluding_padding}; "
          f"As percent accuracy: "
          f"{(total_tokens_considered_in_split_excluding_padding-np.sum(sign_flips_by_sentence))/float(total_tokens_considered_in_split_excluding_padding)}")
    print(f"\tSign flips by sentence (relative to original model): mean: {np.mean(sign_flips_by_sentence)}; "
          f"min: {np.min(sign_flips_by_sentence)};"
          f" max: {np.max(sign_flips_by_sentence)}; std: {np.std(sign_flips_by_sentence)}")

    print(f"\tNearest distances of correct sign (relative to original model): "
          f"mean: {np.mean(nearest_distance_correct_model_sign)}; "
          f"min: {np.min(nearest_distance_correct_model_sign)}; "
          f"max: {np.max(nearest_distance_correct_model_sign)}; "
          f"std: {np.std(nearest_distance_correct_model_sign)}; "
          f"total: {len(nearest_distance_correct_model_sign)}")
    print(f"\tNearest distances of wrong sign (relative to original model): "
          f"mean: {np.mean(nearest_distance_wrong_model_sign)}; "
          f"min: {np.min(nearest_distance_wrong_model_sign)}; "
          f"max: {np.max(nearest_distance_wrong_model_sign)}; "
          f"std: {np.std(nearest_distance_wrong_model_sign)}; "
          f"total: {len(nearest_distance_wrong_model_sign)}")

    if eval_token_level_flips:
        print(f"Total sign flips (relative to ground-truth): {np.sum(true_sign_flips_by_sentence)} out of "
              f"{total_tokens_considered_in_split_excluding_padding}; "
              f"As percent accuracy: "
              f"{(total_tokens_considered_in_split_excluding_padding - np.sum(true_sign_flips_by_sentence)) / float(total_tokens_considered_in_split_excluding_padding)}")
        print(f"\tSign flips by sentence (relative to ground-truth): mean: {np.mean(true_sign_flips_by_sentence)}; "
              f"min: {np.min(true_sign_flips_by_sentence)};"
              f" max: {np.max(true_sign_flips_by_sentence)}; std: {np.std(true_sign_flips_by_sentence)}")

        print(f"\tNearest distances of correct sign (relative to ground-truth): "
              f"mean: {np.mean(nearest_distance_correct_true_sign)}; "
              f"min: {np.min(nearest_distance_correct_true_sign)}; "
              f"max: {np.max(nearest_distance_correct_true_sign)}; "
              f"std: {np.std(nearest_distance_correct_true_sign)}; "
              f"total: {len(nearest_distance_correct_true_sign)}")
        print(f"\tNearest distances of wrong sign (relative to ground-truth): "
              f"mean: {np.mean(nearest_distance_wrong_true_sign)}; "
              f"min: {np.min(nearest_distance_wrong_true_sign)}; "
              f"max: {np.max(nearest_distance_wrong_true_sign)}; "
              f"std: {np.std(nearest_distance_wrong_true_sign)}; "
              f"total: {len(nearest_distance_wrong_true_sign)}")

        if len(nearest_distance_correct_model_sign_and_true_sign) > 0:
            print(f"\tKNN and original model sign match AND *KNN* true ground-truth prediction: "
                  f"mean: {np.mean(nearest_distance_correct_model_sign_and_true_sign)}; "
                  f"min: {np.min(nearest_distance_correct_model_sign_and_true_sign)}; "
                  f"max: {np.max(nearest_distance_correct_model_sign_and_true_sign)}; "
                  f"std: {np.std(nearest_distance_correct_model_sign_and_true_sign)}; "
                  f"total: {len(nearest_distance_correct_model_sign_and_true_sign)}; "
                  f"Accuracy: {len(nearest_distance_correct_model_sign_and_true_sign)/len(nearest_distance_correct_model_sign)}")
        else:
            print(f"\tKNN and original model sign match AND *KNN* true ground-truth prediction: "
                  f"total: {len(nearest_distance_correct_model_sign_and_true_sign)}")

        if len(nearest_distance_wrong_model_sign_and_true_sign) > 0:
            print(f"\tKNN and original model sign DO NOT match AND *KNN* true ground-truth prediction: "
                  f"mean: {np.mean(nearest_distance_wrong_model_sign_and_true_sign)}; "
                  f"min: {np.min(nearest_distance_wrong_model_sign_and_true_sign)}; "
                  f"max: {np.max(nearest_distance_wrong_model_sign_and_true_sign)}; "
                  f"std: {np.std(nearest_distance_wrong_model_sign_and_true_sign)}; "
                  f"total: {len(nearest_distance_wrong_model_sign_and_true_sign)}; "
                  f"Accuracy: {len(nearest_distance_wrong_model_sign_and_true_sign)/len(nearest_distance_wrong_model_sign)}")
        else:
            print(f"\tKNN and original model sign DO NOT match AND *KNN* true ground-truth prediction: "
                  f"total: {len(nearest_distance_wrong_model_sign_and_true_sign)}")

        print(f"Reference total sign flips (relative to ground-truth) of the *original model*: "
              f"{total_sign_flips_of_the_original_model} out of "
              f"{total_tokens_considered_in_split_excluding_padding}; "
              f"As percent accuracy: "
              f"{(total_tokens_considered_in_split_excluding_padding - total_sign_flips_of_the_original_model) / float(total_tokens_considered_in_split_excluding_padding)}")

        # add random and majority class
        # for all tokens:
        print_random_and_majority_class(test_seq_y_for_eval, total_tokens_considered_in_split, flatten_input=True)
        # excluding tokens the original model never saw:
        print_random_and_majority_class(test_seq_y_for_eval_no_padding,
                                        total_tokens_considered_in_split_excluding_padding, flatten_input=False)
        print(f"Random and majority class baselines for approximation to the original model's predictions")
        print_random_and_majority_class(original_model_non_padding_predictions,
                                        total_tokens_considered_in_split_excluding_padding, flatten_input=False)

    print(f"Time to complete eval: {(time.time() - start_eval_time) / 60} minutes")
    eval_stats = utils_eval.calculate_seq_metrics(test_seq_y_for_eval, all_contribution_tuples_by_sentence,
                                     sentence_probs, tune_offset=tune_offset,
                                     print_baselines=False,
                                     output_generated_detection_file=output_generated_detection_file,
                                     numerically_stable=True, fce_eval=True)
    # add flips:
    eval_stats["sign_flips_to_original_model"] = np.sum(sign_flips_by_sentence)
    # Flips to ground-truth are for reference/analysis-only in the zero shot case, as they would not in general
    # be available
    eval_stats["sign_flips_to_ground_truth"] = np.sum(true_sign_flips_by_sentence)
    return eval_stats
    #{"precision": precision, "recall": recall, "fscore_1": fscore_1, "fscore_0_5": fscore_0_5, "mcc": mcc}


def calculate_accuracy(prediction_list, true_list):
    assert len(prediction_list) == len(true_list)
    return np.sum([1 if predicted == true_y else 0 for predicted, true_y in
                         zip(prediction_list, true_list)]) / len(true_list)


def print_random_and_majority_class(seq_y, total_tokens_considered_in_split, flatten_input=True):
    gold_seq_labels_flat = []
    if flatten_input:
        # if the sequence labels are at the sentence level, we flatten them here
        for gold_labels in seq_y:  # data["test_seq_y"]:
            gold_seq_labels_flat.extend([float(one_label) for one_label in gold_labels])
    else:
        gold_seq_labels_flat = seq_y
    assert len(gold_seq_labels_flat) == total_tokens_considered_in_split
    print(
        f"Reference random token-level accuracy ({len(gold_seq_labels_flat)} tokens): "
        f"{calculate_accuracy(np.random.randint(0, high=2, size=len(gold_seq_labels_flat)), gold_seq_labels_flat)}")
    print(
        f"Reference all 1's token-level accuracy ({len(gold_seq_labels_flat)} tokens): "
        f"{calculate_accuracy(np.ones(len(gold_seq_labels_flat)), gold_seq_labels_flat)}")
    print(
        f"Reference all 0's token-level accuracy ({len(gold_seq_labels_flat)} tokens): "
        f"{calculate_accuracy(np.zeros(len(gold_seq_labels_flat)), gold_seq_labels_flat)}")


def save_error_analysis_summary_topk(error_label, model, params, sent_tokens, database_data_structures,
                                     k_nearest_database_indexes,
                                     query_true_token_label,
                                     query_token_contribution, token_i, token_id, sent_i,
                                     contribution_tuples_for_sent, k_nearest_database_distances,
                                     output_exemplar_analysis_file_object, query_non_padding_mask_for_token):
    output_lines = []  # the token-level analysis
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, database_original_sentences, \
        database_sentence_labels = database_data_structures

    output_lines.append(f"{''.join(['-']*45)}")
    output_lines.append(f"{error_label}")
    output_lines.append(f"Query sentence {sent_i}; token index: {token_i}; query true token label: {query_true_token_label}; "
          f"query model logit: {query_token_contribution}, KNN logit: {contribution_tuples_for_sent[token_i][1]}; "
          f"Exceeded max length: {query_non_padding_mask_for_token==0}")
    # original sentence with the token of focus highlighted in brackets
    # query_sentence = string_format_token_list_with_current_token(data["test_sentences"][sent_i], token_i)
    query_sentence = string_format_token_list_with_current_token(sent_tokens, token_i)
    output_lines.append(query_sentence)

    assert k_nearest_database_indexes.shape == k_nearest_database_distances.shape
    output_lines.append(f"Displaying {k_nearest_database_indexes.shape[0]} of {model.top_k} database exemplars")
    # get model terms/weights for reference
    with torch.no_grad():
        term_analysis, additional_model_info = model.get_model_term_decomposition_for_instance(
            k_nearest_database_distances.to(params["main_device"]), k_nearest_database_indexes.to(params["main_device"]))

    output_lines.append(f"Model info: {' '.join(additional_model_info)}")
    for exemplar_index, nearest_database_index, nearest_database_distance in \
            zip(range(k_nearest_database_indexes.shape[0]), k_nearest_database_indexes, k_nearest_database_distances):

        nearest_database_index = nearest_database_index.item()
        nearest_database_distance = nearest_database_distance.item()
        output_lines.append(f"{''.join(['~'] * 25)}")
        output_lines.append(f"Exemplar at k={exemplar_index}")
        # nearest database sentence with the token of focus highlighted in brackets
        database_sent_i = database_token_to_sentence[nearest_database_index]
        database_true_token_label, database_token_contribution = database_token_to_stats[nearest_database_index]
        database_relative_token_i = database_token_to_sent_relative_token_id[nearest_database_index]
        output_lines.append(f"\tDatabase sentence {database_sent_i}; k={exemplar_index} distance: {nearest_database_distance}; "
              f"token index: {database_relative_token_i}; "
              f"database true token label: {database_true_token_label}; "
              f"database model logit: {database_token_contribution}; "
              f"database TRUE sentence label: {database_sentence_labels[database_sent_i]}")
        output_lines.append(f"\tModel term: {' '.join(term_analysis[exemplar_index])}")
        database_sentence = string_format_token_list_with_current_token(
            database_original_sentences[database_sent_i], database_relative_token_i)
        output_lines.append(f"\t{database_sentence}")
    # add newlines and convert to string:
    output_lines = "\n".join(output_lines)+"\n"
    output_exemplar_analysis_file_object.write(output_lines)
    output_exemplar_analysis_file_object.flush()
    return output_exemplar_analysis_file_object


def annotate_sentence_with_token_labels(output_annotations_file_object=None, sent_tokens=None,
                                        knn_prediction_signs=None, token_ids=None, query_token_to_stats=None,
                                        query_non_padding_masks=None):
    sent_tokens_annotated = list(sent_tokens)
    assert len(knn_prediction_signs) == len(sent_tokens_annotated), \
        f"{len(knn_prediction_signs)}, {len(sent_tokens_annotated)}"
    assert len(knn_prediction_signs) == len(token_ids), f"{len(knn_prediction_signs)}, {len(token_ids)}"
    for token_i, token_id in enumerate(token_ids):
        query_true_token_label, query_token_contribution = query_token_to_stats[token_id]
        if query_true_token_label == 1:
            sent_tokens_annotated[token_i] = \
                f"{constants_exa.OUTPUT_ANNOTATION_LABEL_TRUTH}{sent_tokens_annotated[token_i]}"
        if knn_prediction_signs[token_i] == 1:
            sent_tokens_annotated[token_i] = \
                f"{constants_exa.OUTPUT_ANNOTATION_KNN_PREDICTION}{sent_tokens_annotated[token_i]}"
        if query_token_contribution > 0:
            sent_tokens_annotated[token_i] = \
                f"{constants_exa.OUTPUT_ANNOTATION_MODEL_PREDICTION}{sent_tokens_annotated[token_i]}"
        if query_non_padding_masks[token_i] == 0:  # these are tokens that have exceeded the max_length
            sent_tokens_annotated[token_i] = \
                f"{constants_exa.OUTPUT_ANNOTATION_MAX_LENGTH}{sent_tokens_annotated[token_i]}"
            assert query_token_contribution == 0, f"In the original version, padding " \
                                                  f"occurred iff the token contribution " \
                                                  f"was 0. This (and the next assert) " \
                                                  f"can be safely commented " \
                                                  f"out if that convention has changed."
        else:
            assert query_token_contribution != 0
    sent_tokens_annotated = " ".join(sent_tokens_annotated) + "\n"
    output_annotations_file_object.write(sent_tokens_annotated)
    output_annotations_file_object.flush()
    return output_annotations_file_object


def string_format_token_list_with_current_token(original_token_list, token_i, marker_left="[[", marker_right="]]"):
    marked_token = f"{marker_left}{original_token_list[token_i]}{marker_right}"
    reformatted_list = original_token_list[0:token_i] + [marked_token] + original_token_list[token_i+1:]
    return " ".join(reformatted_list)


def get_prediction_summary_topk(error_label, model, params, sent_tokens, database_data_structures,
                                     k_nearest_database_indexes,
                                     query_true_token_label,
                                     query_token_contribution, token_i, token_id, sent_i,
                                     contribution_tuples_for_sent, k_nearest_database_distances,
                                     query_non_padding_mask_for_token):
    # This duplicates some calculations if save_error_analysis_summary_topk() is also called for the token, but this
    # is only run once at inference and is relatively fast, so it's more clear to just keep them separate.
    # The data structures here are a balance between keeping the total size of the file to a minimum while also
    # making it easy to subsequently process analyses from just the single file, so there is some duplication of data
    # that could be accessed via the other archived files.
    # Use get_token_level_stats_from_data_structure() for subsequent accessing.
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, database_original_sentences, \
        database_sentence_labels = database_data_structures
    knn_logit = contribution_tuples_for_sent[token_i][1]
    original_model_logit = query_token_contribution

    assert k_nearest_database_indexes.shape == k_nearest_database_distances.shape
    # get model terms/weights for reference
    with torch.no_grad():
        exemplar_weights_list, model_bias = model.get_model_term_decomposition_for_instance(
            k_nearest_database_distances.to(params["main_device"]), k_nearest_database_indexes.to(params["main_device"]),
            as_string=False)
    db_indexes = []  # global token_id's in the database
    db_distances = []
    db_true_token_labels = []
    db_true_sentence_labels = []
    db_original_model_logits = []

    for exemplar_index, nearest_database_index, nearest_database_distance in \
            zip(range(k_nearest_database_indexes.shape[0]), k_nearest_database_indexes, k_nearest_database_distances):

        nearest_database_index = nearest_database_index.item()
        nearest_database_distance = nearest_database_distance.item()

        # nearest database sentence with the token of focus
        database_sent_i = database_token_to_sentence[nearest_database_index]
        database_true_token_label, database_token_contribution = database_token_to_stats[nearest_database_index]
        database_relative_token_i = database_token_to_sent_relative_token_id[nearest_database_index]

        db_indexes.append(nearest_database_index)
        db_distances.append(nearest_database_distance)
        db_true_token_labels.append(database_true_token_label)
        db_true_sentence_labels.append(database_sentence_labels[database_sent_i])
        db_original_model_logits.append(database_token_contribution)

    # note that token_id is the global token index -- only included for debugging/checking
    token_level_stats = [token_id, query_true_token_label, original_model_logit, knn_logit,
                         query_non_padding_mask_for_token]
    # These appends are just to highlight that the following are all lists of size K
    token_level_stats.append(exemplar_weights_list)
    token_level_stats.append(db_indexes)
    token_level_stats.append(db_distances)
    token_level_stats.append(db_true_token_labels)
    token_level_stats.append(db_true_sentence_labels)
    token_level_stats.append(db_original_model_logits)
    # the KNN model bias is constant across tokens, so is saved at the sentence level instead of here
    return token_level_stats


def get_token_level_stats_from_data_structure(token_level_stats):
    # named_stats = {
    #     "token_id": int: global token_id for the query,
    #     "query_true_token_label": int: {0,1}: true token label for the query,
    #     "original_model_logit": float: [-inf, inf]: token-level prediction from the original CNN decomposition,
    #     "knn_logit": float: [-inf, inf]: prediction from the KNN for the token,
    #     "query_non_padding_mask_for_token": 0 if token exceeded max length of original model, else 1
    #   THE FOLLOWING ARE ALL LISTS OF SIZE K
    #     "exemplar_weights_list": list of prediction floats for each of the K exemplars; summing across these and
    #                              adding the KNN model bias should equal 'knn_logit',
    #     "db_indexes": global token_id's for the K matched database exemplars,
    #     "db_distances": L2 distances between the query and the nearest K matched database exemplars,
    #     "db_true_token_labels": true token-level labels for the K matched database exemplars,
    #     "db_true_sentence_labels": true sentence-level labels for the K matched database exemplars,
    #     "db_original_model_logits": the original model logits (via the original CNN decomposition)
    #                                 for the K matched database exemplars,
    # }

    named_stats = {
        "token_id": token_level_stats[0],
        "query_true_token_label": token_level_stats[1],
        "original_model_logit": token_level_stats[2],
        "knn_logit": token_level_stats[3],
        "query_non_padding_mask_for_token": token_level_stats[4],
        "exemplar_weights_list": token_level_stats[5],
        "db_indexes": token_level_stats[6],
        "db_distances": token_level_stats[7],
        "db_true_token_labels": token_level_stats[8],
        "db_true_sentence_labels": token_level_stats[9],
        "db_original_model_logits": token_level_stats[10],
    }
    return named_stats

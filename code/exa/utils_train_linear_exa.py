# Train the K-NN from exemplar distances against the support set.

from model_exa import LinearExA
import utils_exemplar
import utils_linear_exa
import utils_eval_linear_exa
import utils_eval
import utils
import constants
import constants_exa

import numpy as np
import codecs
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import subprocess
import time

from sklearn.utils import shuffle


def update_and_display_losses(cumulative_losses, all_epoch_cumulative_losses, label):
    print(f"Epoch average loss ({label}): {np.mean(cumulative_losses)}")
    all_epoch_cumulative_losses.extend(cumulative_losses)
    print(f"Average loss across all mini-batches (all epochs) ({label}): {np.mean(all_epoch_cumulative_losses)}")
    return all_epoch_cumulative_losses


def display_model_weights_summary(model, label, options):
    print(f"Model ({model.model_type}) weights summary ({label}):")
    print(f"\tmodel.model_bias: {model.model_bias.item()}")
    print(f"\tmodel.model_gamma (y_n): {model.model_gamma.item()}")

    if model.model_type in [constants_exa.MODEL_KNN_MAXENT, constants_exa.MODEL_KNN_LEARNED_WEIGHTING]:
        print(f"\tmodel.model_temperature: {model.model_temperature.item()}")

    if model.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
        with torch.no_grad():
            print(f"\tmodel.model_support_weights (w_k for K): {model.model_support_weights.data}")
            w_k = model.softmax_with_temperature(model.model_support_weights.data)
            print(f"\tmodel.model_support_weights (minent(w_k) for K): {w_k}")
            print(f"\tmodel.model_support_weights (minent(w_k) for K) as "
                  f"w_k/{1/(model.model_support_weights.data.shape[0])}: "
                  f"{w_k/(1/(model.model_support_weights.data.shape[0]))}")


def update_model_database(model, options, database_predicted_contributions,
                          database_true_sentence_level_labels=None, database_true_labels=None):
    model.update_model_prediction_logits(database_predicted_contributions)
    if options.use_sentence_level_database_ground_truth:
        assert not options.use_token_level_database_ground_truth
        print("The ExA model has access to the database true sentence-level labels, but NOT the token-level labels.")
        model.update_model_true_labels(database_true_sentence_level_labels)
    elif options.use_token_level_database_ground_truth:
        assert not options.use_sentence_level_database_ground_truth
        print(f"--WARNING--WARNING--WARNING--")
        print("The ExA model has access to the database true *token*-level labels. NOTE THIS IS ONLY REALISTIC FOR "
              "THE SUPERVISED SCENARIO *AND/OR* -UPDATING- THE DATABASE WITH TOKEN-LEVEL LABELS.")
        model.update_model_true_labels(database_true_labels)
    else:
        assert False
    return model


def get_database_meta_data_for_model(params, database_sentence_to_stats, database_token_to_sentence,
                                     database_token_to_stats):
    total_database_tokens = len(database_token_to_stats)
    print(f"Constructing database for model")
    print(f"Total number of database tokens: {total_database_tokens}")
    # the true sentence-level label for each token (converted to {-1,1}; vector indexed by token id, where
    # each token in the sentence has the same label; INITIALLY IN {0,1} IN options.database_data_structure_file
    database_true_sentence_level_labels = torch.zeros(total_database_tokens).to(params["main_device"])
    # the true label for each token (converted to {-1,1}; vector indexed by token id
    # Note that these should only be used in the fully supervised case (or when updating the database with
    # token-level annotations)
    database_true_labels = torch.zeros(total_database_tokens).to(params["main_device"])
    # the predicted label for each token; vector indexed by token id; not currently
    # used by the model; hard threshold in {-1,1} of contribution logits
    # database_predicted_labels = torch.zeros(total_database_tokens).to(params["main_device"])
    # the contribution logits (i.e., the token-level prediction logits of the original model)
    database_predicted_contributions = torch.zeros(total_database_tokens).to(params["main_device"])

    # # sentence level labels in the original [0,1] domain; used as a mask
    # database_true_sentence_level_labels01 = torch.zeros(total_database_tokens).to(params["main_device"])

    for token_id in range(total_database_tokens):
        database_true_token_label, database_token_contribution = database_token_to_stats[token_id]
        if database_true_token_label == 0:
            database_true_labels[token_id] = -1
        else:
            database_true_labels[token_id] = 1
        database_predicted_contributions[token_id] = database_token_contribution
        # if database_token_contribution > 0:
        #     database_predicted_labels[token_id] = 1
        # else:
        #     database_predicted_labels[token_id] = -1
        sent_i = database_token_to_sentence[token_id]
        true_sentence_level_label = database_sentence_to_stats[sent_i][0]
        # the raw value is expected to be in 0,1
        assert true_sentence_level_label in [0, 1]
        database_true_sentence_level_labels[token_id] = 1 if true_sentence_level_label == 1 else -1
        assert database_true_sentence_level_labels[token_id] in [-1, 1]
        # the intuition here is that for these tasks, if the sentence level label is 0, then the tokens should all be 0
        # database_true_sentence_level_labels01[token_id] = true_sentence_level_label
    return total_database_tokens, database_true_sentence_level_labels, database_true_labels, database_predicted_contributions


def update_database_of_existing_model(model, params, options, database_data_structures):
    # Use this to update the database meta structure of an already trained K-NN model
    print(f"Updating the trained K-NN model's database")
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, _, _ = database_data_structures

    total_database_tokens, database_true_sentence_level_labels, database_true_labels, \
        database_predicted_contributions = get_database_meta_data_for_model(params,
                                                                            database_sentence_to_stats,
                                                                            database_token_to_sentence,
                                                                            database_token_to_stats)
    if options.zero_original_model_logits_not_matching_label_sign:
        # Note that these are primarily only for research purposes and debugging. If this results in significantly
        # different effectiveness, it may be a sign that the model is a particularly poor predictor over the
        # instances in the updated database and thus caution would be in order in using the resulting model.
        if options.use_sentence_level_database_ground_truth:
            print(f"WARNING: The original model's prediction logits are being zero'd if they do not match the "
                  f"sign of the ground-truth SENTENCE-level labels if the sentence-level label is 0.")
            class0_sign_mask = database_true_sentence_level_labels.eq(-1)  # assumes class 0 is -1
            # only 0's for class 0 with sign mismatches; padding is unaffected since here the logit is always 0
            # for padding
            same_sign_mask = torch.eq(torch.sign(class0_sign_mask*database_predicted_contributions),
                                      torch.sign(class0_sign_mask*database_true_sentence_level_labels))
        elif options.use_token_level_database_ground_truth:
            print(f"WARNING: The original model's prediction logits are being zero'd if they do not match the "
                  f"sign of the ground-truth TOKEN-level labels.")
            same_sign_mask = torch.eq(torch.sign(database_predicted_contributions),
                                          torch.sign(database_true_labels))
        else:
            assert False
        database_predicted_contributions = same_sign_mask*database_predicted_contributions
    model = update_model_database(model, options, database_predicted_contributions,
                                  database_true_sentence_level_labels=database_true_sentence_level_labels,
                                  database_true_labels=database_true_labels)
    return model


def train_linear_exa(params, options, data, number_of_filter_maps, output_generated_detection_file, np_random_state):

    start_train_time = time.time()

    #max_metric_label = "fscore_0_5"
    max_metric_label = options.max_metric_label

    print(f'Initial allocation (note BERT is currently loaded) torch.cuda.max_memory_allocated: '
          f'{torch.cuda.max_memory_allocated(params["main_device"])}')

    #y = data["test_y"]
    print(f"Loading exemplar data.")
    database_data_structures = \
        utils_linear_exa.load_data_structure_torch_from_file(options.database_data_structure_file, -1)
    print(f"Loaded database data structure file from {options.database_data_structure_file}")
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, _, _ = database_data_structures

    total_database_tokens, database_true_sentence_level_labels, database_true_labels, \
        database_predicted_contributions = get_database_meta_data_for_model(params,
                                                                            database_sentence_to_stats,
                                                                            database_token_to_sentence,
                                                                            database_token_to_stats)

    print(f'database distances additional structures torch.cuda.max_memory_allocated: '
          f'{torch.cuda.max_memory_allocated(params["main_device"])}')
    query_data_structures = \
        utils_linear_exa.load_data_structure_torch_from_file(options.query_data_structure_file, -1)
    query_sentence_to_stats, query_sentence_to_tokens, query_token_to_sentence, \
        query_token_to_sent_relative_token_id, query_token_to_stats, query_non_padding_masks_by_sentence = \
        query_data_structures
    assert len(query_sentence_to_stats) == len(data["test_sentences"])
    assert len(query_non_padding_masks_by_sentence) == len(data["test_sentences"])
    print(f"Total number of query tokens: {len(query_token_to_stats)}")

    eval_data_structures = \
        utils_linear_exa.load_data_structure_torch_from_file(options.query_data_structure_file, -1)

    if options.restrict_eval_to_query_eval_split_chunk_ids_file:
        chunk_ids = \
            utils_linear_exa.load_data_structure_torch_from_file(options.query_train_split_chunk_ids_file, -1).numpy()
        print(f"Training on the subset of data specified by {options.query_train_split_chunk_ids_file}")
        eval_split_chunk_ids_subset = \
            utils_linear_exa.load_data_structure_torch_from_file(options.query_eval_split_chunk_ids_file, -1).numpy()
        print(f"Evaluating on the subset of data specified by {options.query_eval_split_chunk_ids_file}")
    else:
        print(f"Training and evaluating on all chunk_ids in {options.distance_dir}")
        chunk_ids = utils_linear_exa.load_memory_structure_torch(options.distance_dir,
                                                                f"distances_to_db_chunk_ids", 0, -1).numpy()

    # get the query logits for training: these are the prediction targets \hat{y_N+1} at the *token* level grouped
    # by the sentence chunks
    hard_target_threshold = False
    if hard_target_threshold:
        print(f"Hard threshold on targets.")
    else:
        print(f"Soft threshold on targets.")
    query_predicted_contributions_by_chunks = []
    query_non_padding_by_chunks = []
    for chunk_id in chunk_ids:
        sentence_ids_in_current_chunk = utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db_sentence_ids",
                                                                         chunk_id, -1).numpy()
        query_token_logits_in_chunk = []
        query_non_padding_in_chunk = []
        for sent_i in sentence_ids_in_current_chunk:
            token_ids = query_sentence_to_tokens[sent_i]
            # all of the padding masks for the sentence
            query_non_padding_masks_for_sent_i = query_non_padding_masks_by_sentence[sent_i]
            assert len(query_non_padding_masks_for_sent_i) == len(token_ids)
            query_non_padding_in_chunk.extend(query_non_padding_masks_for_sent_i)
            for token_i, token_id in enumerate(token_ids):
                _, query_token_contribution = query_token_to_stats[token_id]
                if query_token_contribution == 0:  # padding check; see comment in first assert
                    assert query_non_padding_masks_for_sent_i[token_i] == 0, f"In the original version, padding " \
                                                                             f"occurred iff the token contribution " \
                                                                             f"was 0. This (and the next assert) " \
                                                                             f"can be safely commented " \
                                                                             f"out if that convention has changed."
                else:
                    assert query_non_padding_masks_for_sent_i[token_i] == 1
                if not hard_target_threshold:
                    query_token_logits_in_chunk.append(query_token_contribution)
                else:
                    if query_token_contribution > 0:
                        query_token_logits_in_chunk.append(1.0)
                    else:
                        query_token_logits_in_chunk.append(0.0)
        query_predicted_contributions_by_chunks.append(query_token_logits_in_chunk)
        query_non_padding_by_chunks.append(query_non_padding_in_chunk)

    print(f"Exemplar training data loaded.")

    linear_exa_params = {"support_size": total_database_tokens,
                         "top_k": options.top_k,
                         "model_type": options.model_type,
                         "model_temperature_init_value": float(options.model_temperature_init_value),
                         "model_support_weights_init_values":
                             [float(x) for x in options.model_support_weights_init_values.split(",")],
                         "model_bias_init_value": float(options.model_bias_init_value),
                         "model_gamma_init_value": float(options.model_gamma_init_value)}

    print(f"Initializing model with params: {linear_exa_params} and original support logits")
    model = LinearExA(**linear_exa_params).cuda(params["GPU"])
    model = update_model_database(model, options, database_predicted_contributions,
                                  database_true_sentence_level_labels=database_true_sentence_level_labels,
                                  database_true_labels=database_true_labels)

    print(f'model init torch.cuda.max_memory_allocated: '
          f'{torch.cuda.max_memory_allocated(params["main_device"])}')
    print("Starting training")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if options.use_sgd:
        optimizer = optim.SGD(parameters, lr=options.learning_rate)
        print(
            f"Using optim.SGD with LR: {options.learning_rate} with BCEWithLogitsLoss and IGNORING grad clip max_norm={params['NORM_LIMIT']}")
    else:
        optimizer = optim.Adadelta(parameters, options.learning_rate)
        print(
            f"Using optim.Adadelta with LR: {options.learning_rate} with BCEWithLogitsLoss and IGNORING grad clip max_norm={params['NORM_LIMIT']}")
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    display_model_weights_summary(model, f"before training", options)

    max_metric_score = init_max_metric_score(max_metric_label)
    print(f"Max metric score initialized to {max_metric_score}.")
    max_metric_epoch = -1
    all_epoch_cumulative_losses = []

    if model.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
        k_support_constraint_target = torch.ones(options.top_k-1+2).to(params["main_device"])
        k_support_constraint_target[-2] = 0.0  # for the min

    for e in range(params["EPOCH"]):
        print(f"Starting epoch {e}")
        start_epoch_time = time.time()
        cumulative_losses = []
        if e != 0 and max_metric_epoch != e - 1 and not options.never_mask_sign_matches:
            print(f"\tPrevious epoch is not an improvement, so only considering sign mis-matches in the loss this epoch ({e}).")
        # loop through query sentences
        print(f"Shuffling data")
        chunk_ids, query_predicted_contributions_by_chunks, query_non_padding_by_chunks = \
            shuffle(chunk_ids, query_predicted_contributions_by_chunks, query_non_padding_by_chunks,
                    random_state=np_random_state)

        for chunk_process_index, chunk_id in enumerate(chunk_ids):
            print(f"TopK: Processing distance chunk index {chunk_process_index} of {len(chunk_ids)}: chunk id {chunk_id}.")
            distances_by_sentences_chunk = \
                utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db",
                                                             chunk_id, int(params["GPU"]))
            support_set_indexes_by_sentences_chunk = \
                utils_linear_exa.load_memory_structure_torch(options.distance_dir, f"distances_to_db_support_indexes",
                                                             chunk_id, int(params["GPU"]))

            # if chunk_id == 0:
            #     print(f'distance structures for chunk 0 torch.cuda.max_memory_allocated: '
            #           f'{torch.cuda.max_memory_allocated(params["main_device"])}')

            query_predicted_contributions = torch.FloatTensor(query_predicted_contributions_by_chunks[chunk_process_index]).to(params["main_device"])
            if not options.never_mask_beyond_max_length_padding:
                non_padding_mask = torch.FloatTensor(query_non_padding_by_chunks[chunk_process_index]).to(params["main_device"])
            optimizer.zero_grad()
            model.train()

            if model.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
                predicted_token_contributions, k_support_weights = \
                    model.forward(top_k_distances=distances_by_sentences_chunk,
                                  top_k_distances_idx=support_set_indexes_by_sentences_chunk)
                # want to encourage sparse weights of decreasing magnitude as k increases:
                # Specifically, want w_k-w_{k+1} > 0 and max to be high, min to be low
                # Note that this operates on the logits, which get transformed to a distribution in [0,1] in the forward
                support_diff = k_support_weights[0:-1] - k_support_weights[1:]
                support_weight_max, _ = torch.max(k_support_weights, dim=0)
                if options.randomize_min_index_limit:
                    # randomize the end index of weights to consider for the min; otherwise, may always be final weight
                    # due to the decreasing constraint; the first half of the weights are excluded; also, note that
                    # the weight initialization is not not important; this may be particularly helpful for large K
                    support_index_limit = torch.randint(low=options.top_k // 2, high=options.top_k, size=(1,))
                    # we exclude the first weight, which would typically be the max in this scheme in any case (this
                    # is just a reminder that the support_index_limit should never be such that only the first weight
                    # is considered for the min)
                    support_weight_min, _ = torch.min(k_support_weights[1:support_index_limit + 1], dim=0)
                else:
                    # otherwise, all support weights are considered for min and max
                    support_weight_min, _ = torch.min(k_support_weights, dim=0)
                # note there should be options.top_k-1 differences (no lower bound on the final weight)
                min_max_weights_tensor = torch.zeros(options.top_k - 1 + 2).to(params["main_device"])
                min_max_weights_tensor[0:options.top_k - 1] = support_diff
                min_max_weights_tensor[-2] = support_weight_min
                min_max_weights_tensor[-1] = support_weight_max

                loss_support_weights = criterion(min_max_weights_tensor, k_support_constraint_target).mean()
            else:
                predicted_token_contributions = \
                    model.forward(top_k_distances=distances_by_sentences_chunk,
                                  top_k_distances_idx=support_set_indexes_by_sentences_chunk)

            loss = criterion(predicted_token_contributions, torch.sigmoid(query_predicted_contributions))
            with torch.no_grad():
                # for reference, note that torch.sign(torch.tensor([0.0])) == 0 but OK here since we only care about !=
                # and 0.0 is masked anyway since it occurs with padding
                not_same_sign_mask = torch.ne(torch.sign(predicted_token_contributions), torch.sign(query_predicted_contributions))

            if e != 0 and max_metric_epoch != e-1 and not options.never_mask_sign_matches:  # only mask when the previous epoch is not the top
                # only consider loss for sign mis-matches:
                if not options.never_mask_beyond_max_length_padding:
                    loss = (non_padding_mask*not_same_sign_mask * loss).mean()
                else:
                    loss = (not_same_sign_mask * loss).mean()
            else:
                if not options.never_mask_beyond_max_length_padding:
                    loss = (non_padding_mask*loss).mean()
                else:
                    loss = loss.mean()
            if model.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
                # matching output more important than the weight constraints, so rescale latter by length:
                loss = (loss + (1/min_max_weights_tensor.shape[0]) * loss_support_weights) / 2
                #loss = (loss + loss_support_weights) / 2

            cumulative_losses.append(loss.item())
            loss.backward()
            #nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

            if chunk_process_index == 0 and e == 0:
                nvidia_smi_out = subprocess.run(["nvidia-smi"], capture_output=True)
                print(f'{nvidia_smi_out.stdout.decode("utf-8")}')

        print(f"Time to complete epoch: {(time.time() - start_epoch_time) / 60} minutes")
        print(f"Cumulative overall training time: {(time.time() - start_train_time) / 60} minutes")
        # print weights
        update_and_display_losses(cumulative_losses, all_epoch_cumulative_losses, f"epoch {e}")
        display_model_weights_summary(model, f"epoch {e}", options)

        if not options.restrict_eval_to_query_eval_split_chunk_ids_file:
            print(f"Beginning eval on the full split")
            eval_stats = utils_eval_linear_exa.eval_linear_exa(eval_data_structures, model, params, options, data,
                                                               number_of_filter_maps, output_generated_detection_file)
            print(f"Summary scores: Epoch {e}: max_{max_metric_label}")
        else:
            print(f"{''.join(['~'] * 25)}Beginning eval on train (epoch {e}){''.join(['~'] * 25)}")
            train_eval_stats = utils_eval_linear_exa.eval_linear_exa(eval_data_structures, model, params, options, data,
                                                                     number_of_filter_maps,
                                                                     output_generated_detection_file,
                                                                     database_data_structures=None,
                                                                     eval_split_chunk_ids_subset=chunk_ids,
                                                                     tune_offset=False,
                                                                     print_memory_stats=False)
            print(f"{''.join(['~'] * 25)}Beginning eval on dev split (epoch {e}){''.join(['~'] * 25)}")
            eval_stats = utils_eval_linear_exa.eval_linear_exa(eval_data_structures, model, params, options, data,
                                                               number_of_filter_maps,
                                                               output_generated_detection_file,
                                                               database_data_structures=None,
                                                               eval_split_chunk_ids_subset=eval_split_chunk_ids_subset,
                                                               tune_offset=False,
                                                               print_memory_stats=False)
            print(f"Summary scores: Epoch {e}: max_{max_metric_label} (train): {train_eval_stats[max_metric_label]}; "
                  f"max_{max_metric_label} (dev): {eval_stats[max_metric_label]}")

        if max_metric_comparator(eval_stats, max_metric_label, max_metric_score):
            max_metric_score = eval_stats[max_metric_label]
            max_metric_epoch = e
            print(
                f"Saving epoch {e} as new best max_{max_metric_label}_epoch model with score of {max_metric_score}")
            utils.save_model_torch(model, params, f"best_max_{max_metric_label}_epoch")

        print(f"\tCurrent max dev {max_metric_label} score: {max_metric_score} at epoch {max_metric_epoch}")


def max_metric_comparator(eval_stats, max_metric_label, max_metric_score):
    if max_metric_label in ["sign_flips_to_original_model", "sign_flips_to_ground_truth"]:
        # use min
        return eval_stats[max_metric_label] <= max_metric_score
    else:
        return eval_stats[max_metric_label] >= max_metric_score


def init_max_metric_score(max_metric_label):
    if max_metric_label in ["sign_flips_to_original_model", "sign_flips_to_ground_truth"]:
        return np.inf
    else:
        return -np.inf

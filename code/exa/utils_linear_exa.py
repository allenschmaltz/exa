import utils_exemplar
import constants

import numpy as np
import codecs
from collections import defaultdict
import torch
import torch.nn as nn


def save_memory_structure_torch(memory_dir, file_identifier_prefix, memory_structure, chunk_id):

    path = f"{memory_dir}/{file_identifier_prefix}_memory_{chunk_id}.pt"
    torch.save(memory_structure, path)


def load_memory_structure_torch(memory_dir, file_identifier_prefix, chunk_id, onto_gpu_id):
    path = f"{memory_dir}/{file_identifier_prefix}_memory_{chunk_id}.pt"
    try:
        if onto_gpu_id != -1:
            memory_structure = torch.load(path, map_location=lambda storage, loc: storage.cuda(onto_gpu_id))
        else:
            memory_structure = torch.load(path, map_location=lambda storage, loc: storage)
        return memory_structure
    except:
        print(f"No available memory structure at {path}.")


def save_data_structure_torch_to_file(path_with_filename, data_structure):

    torch.save(data_structure, path_with_filename)


def load_data_structure_torch_from_file(path_with_filename, onto_gpu_id):
    try:
        if onto_gpu_id != -1:
            data_structure = torch.load(path_with_filename, map_location=lambda storage, loc: storage.cuda(onto_gpu_id))
        else:
            data_structure = torch.load(path_with_filename, map_location=lambda storage, loc: storage)
        return data_structure
    except:
        print(f"No available data structure at {path_with_filename}.")


def save_exemplar_distances(params, options, data, number_of_filter_maps):

    pdist = nn.PairwiseDistance(p=2)
    y = data["test_y"]
    print(f"Loading exemplar data.")
    database_sentence_to_stats, database_sentence_to_tokens, database_token_to_sentence, \
        database_token_to_sent_relative_token_id, database_token_to_stats, database_token_by_unicnn_filter, \
        database_mask_sent_true_class_0, database_mask_sent_true_class_1 = \
        utils_exemplar.get_exemplar_data_from_file(options.exemplar_data_database_file, number_of_filter_maps[0])

    total_database_tokens = len(database_token_to_stats)
    print(f"Total number of database tokens: {total_database_tokens}")

    query_sentence_to_stats, query_sentence_to_tokens, query_token_to_sentence, query_token_to_sent_relative_token_id, \
        query_token_to_stats, query_token_by_unicnn_filter, _, _ = \
        utils_exemplar.get_exemplar_data_from_file(options.exemplar_data_query_file, number_of_filter_maps[0])

    print(f"Total number of query tokens: {len(query_token_to_stats)}")
    print(f"Exemplar data loaded.")

    # Currently we load the entire DB onto the GPU. The distances calculations are independent, so this can be split
    # if it exceeds your GPU's memory: Just run a loop over subsets of the database and then concat the distances.
    database_token_by_unicnn_filter = torch.FloatTensor(database_token_by_unicnn_filter).to(params["main_device"])
    query_token_by_unicnn_filter = torch.FloatTensor(query_token_by_unicnn_filter).to(params["main_device"])
    print(f"Placed database and query on device")

    assert options.distance_sentence_chunk_size > 0
    distances_by_sentences_chunk = []  # options.distance_sentence_chunk_size
    # these are the indexes in the support set corresponding to the top k distances in distances_by_sentences_chunk:
    support_set_indexes_by_sentences_chunk = []
    sentence_ids_in_current_chunk = []
    chunk_ids = []
    chunk_id = 0
    # loop through query sentences
    assert len(query_sentence_to_stats) == len(data["test_sentences"])
    for sent_i, sent_tokens in enumerate(data["test_sentences"]):
        assert len(sent_tokens) == len(query_sentence_to_tokens[sent_i])
        assert query_sentence_to_stats[sent_i][0] == y[sent_i]
        if sent_i % options.distance_sentence_chunk_size == 0:
            print(f"Currently processing sentence id {sent_i}")
        token_ids = query_sentence_to_tokens[sent_i]
        sentence_ids_in_current_chunk.append(sent_i)
        for sent_relative_token_id, token_id in enumerate(token_ids):
            one_query_token_by_unicnn_filter = query_token_by_unicnn_filter[
                query_sentence_to_tokens[sent_i][sent_relative_token_id]]
            assert query_sentence_to_tokens[sent_i][sent_relative_token_id] == token_id
            l2_pairwise_distances = pdist(one_query_token_by_unicnn_filter.expand_as(database_token_by_unicnn_filter),
                                          database_token_by_unicnn_filter)
            #distances_by_sentences_chunk.append(l2_pairwise_distances.unsqueeze(0))
            top_k_distances, top_k_distances_idx = torch.topk(l2_pairwise_distances, options.top_k, largest=False,
                                                              sorted=True)
            distances_by_sentences_chunk.append(top_k_distances.unsqueeze(0))
            support_set_indexes_by_sentences_chunk.append(top_k_distances_idx.unsqueeze(0))

        # we save by sentence (to facilitate later shuffling for training/eval), but note that the rows are the tokens
        if len(sentence_ids_in_current_chunk) == options.distance_sentence_chunk_size:
            save_memory_structure_torch(options.distance_dir, f"distances_to_db",
                                        torch.cat(distances_by_sentences_chunk, 0), chunk_id)
            save_memory_structure_torch(options.distance_dir, f"distances_to_db_support_indexes",
                                        torch.cat(support_set_indexes_by_sentences_chunk, 0), chunk_id)
            save_memory_structure_torch(options.distance_dir, f"distances_to_db_sentence_ids",
                                        torch.LongTensor(sentence_ids_in_current_chunk), chunk_id)
            chunk_ids.append(chunk_id)
            chunk_id += 1
            distances_by_sentences_chunk = []
            support_set_indexes_by_sentences_chunk = []
            sentence_ids_in_current_chunk = []

    # any remaining
    if len(sentence_ids_in_current_chunk) != 0:
        assert len(distances_by_sentences_chunk) != 0
        save_memory_structure_torch(options.distance_dir, f"distances_to_db",
                                    torch.cat(distances_by_sentences_chunk, 0), chunk_id)
        save_memory_structure_torch(options.distance_dir, f"distances_to_db_support_indexes",
                                    torch.cat(support_set_indexes_by_sentences_chunk, 0), chunk_id)
        save_memory_structure_torch(options.distance_dir, f"distances_to_db_sentence_ids",
                                    torch.LongTensor(sentence_ids_in_current_chunk), chunk_id)
        chunk_ids.append(chunk_id)
        # chunk_id += 1
        # distances_by_sentences_chunk = []
        # support_set_indexes_by_sentences_chunk = []
        # sentence_ids_in_current_chunk = []

    save_memory_structure_torch(options.distance_dir, f"distances_to_db_chunk_ids", torch.LongTensor(chunk_ids), 0)
    print(f"The top {options.top_k} distances structures successfully saved to {options.distance_dir}")


def get_exemplar_data_only_stats_from_file(filepath_with_name, expected_filter_size):
    # This is similar to utils_exemplar.get_exemplar_data_from_file() but the filter values are not returned

    # Note that mask_sent_true_class_0, mask_sent_true_class_1 correspond to true sentence-level labels, so they can be
    # used by the database but typically not for the query
    sentence_to_stats = {}  # sent id -> true sentence-level label, sentence neg logit, sentence pos logit, neg logit bias, pos logit bias
    token_to_sentence = {}  # token id (i.e., row in token_by_unicnn_filter) -> corresponding sentence id
    token_to_sent_relative_token_id = {}  # token id -> corresponding index into the corresponding sentence
    sentence_to_tokens = defaultdict(list)  # sent id -> list of token ids
    token_to_stats = {}  # token id -> true token label, token contribution
    token_by_unicnn_filter = []  # list of filter values -- here, just a placeholder int to analogously track the token_id
    sent_id = -1
    sent_relative_token_id = 0
    mask_sent_true_class_0 = []  # 1's for sentence-level class 0 (true label); else float("inf")
    mask_sent_true_class_1 = []  # 1's for sentence-level class 1 (true label); else float("inf")
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split(",")
            if tokens[0].startswith("SENT"):
                assert len(tokens) == 6
                assert tokens[0] not in sentence_to_stats
                sent_id = int(tokens[0][len("SENT"):])
                sentence_to_stats[sent_id] = [float(x) for x in tokens[1:]]
                # for consistency, the sentence-level label (as with the token-level label) is an int:
                sentence_to_stats[sent_id][0] = int(sentence_to_stats[sent_id][0])
                sent_relative_token_id = 0
            else:
                assert len(tokens) == 2 + expected_filter_size
                tokens = [float(x) for x in tokens]
                true_token_label = int(tokens[0])
                token_contribution = tokens[1]
                unicnn_filter = 1  # just a placeholder #tokens[2:]
                token_id = len(token_by_unicnn_filter)
                token_by_unicnn_filter.append(unicnn_filter)
                assert sent_id != -1
                assert token_id not in token_to_sentence
                assert token_id not in token_to_stats
                token_to_sentence[token_id] = sent_id
                token_to_sent_relative_token_id[token_id] = sent_relative_token_id
                sent_relative_token_id += 1
                sentence_to_tokens[sent_id].append(token_id)
                token_to_stats[token_id] = [true_token_label, token_contribution]
                # update masks:
                if sentence_to_stats[sent_id][0] == 0:
                    mask_sent_true_class_0.append(1)
                    mask_sent_true_class_1.append(float("inf"))
                elif sentence_to_stats[sent_id][0] == 1:
                    mask_sent_true_class_0.append(float("inf"))
                    mask_sent_true_class_1.append(1)
                else:
                    assert False

    return sentence_to_stats, sentence_to_tokens, token_to_sentence, token_to_sent_relative_token_id, token_to_stats, \
           token_by_unicnn_filter, mask_sent_true_class_0, mask_sent_true_class_1


def get_real_token_non_padding(data, max_length):
    # Create non-padding masks (1 for a real token, 0 for padding). This is intended to be used with query
    # sentences, primarily for comparing sign flips to the original model, since it's not meaningful to compare against
    # tokens the original model never saw. In principle with the current models, the end-of-sentence padding tokens
    # are uniquely described by query_token_contribution == 0, but it is safer to use this structure, in case
    # the calculation of the contribution ever changes. Note that it is not valid to just use max_length,
    # since the original model used BERT WordPieces, which get de-tokenized in constructing the exemplars in order
    # to evaluate against the original sequence-level labels, which are at the 'word' level.

    non_padding_masks_by_sentence = []

    all_untruncated_tokenized_tokens, all_original_untokenized_tokens = data["test_x"], data["test_sentences"]
    all_bert_to_original_tokenization_maps = data["test_bert_to_original_tokenization_maps"]
    assert len(all_untruncated_tokenized_tokens) == len(all_original_untokenized_tokens)
    assert len(all_untruncated_tokenized_tokens) == len(all_bert_to_original_tokenization_maps)
    # BERT de-tokenization exactly matches that of initial exemplar vector construction:
    print(f"Creating non-padding masks for {len(all_original_untokenized_tokens)} sentences.")
    for sentence_index in range(len(all_original_untokenized_tokens)):
        untruncated_tokenized_tokens = all_untruncated_tokenized_tokens[sentence_index]
        # the original tokens (without additional tokenization) aligned with the gold labels
        original_untokenized_tokens = all_original_untokenized_tokens[sentence_index]

        # training_observed_length is the total number of real (tokenized) tokens seen by the model for this instance,
        # ignoring prefix/suffix padding:
        training_observed_length = min(max_length, len(untruncated_tokenized_tokens))
        non_padding_mask = [1.0] * training_observed_length
        non_padding_mask.extend([0.0]*(len(untruncated_tokenized_tokens) - max_length))

        bert_to_original_tokenization_map = all_bert_to_original_tokenization_maps[sentence_index]
        detokenized_non_padding_mask_dict = defaultdict(list)  # keys are the original token indexes
        for bert_to_token, padding_mask in zip(bert_to_original_tokenization_map, non_padding_mask):
            detokenized_non_padding_mask_dict[bert_to_token].append(padding_mask)

        detokenized_non_padding_mask = []
        for original_token_id in range(len(detokenized_non_padding_mask_dict)):
            # Some original tokens were split by BERT, so the generated masks need to be combined. Here, we
            # only consider a token to be a padding symbol if *all* the WordPieces were padding. As such,
            # it is sufficient to just take the max padding mask of all of the WordPieces, since non-padding is
            # 1.0.
            word_piece_masks = detokenized_non_padding_mask_dict[original_token_id]
            detokenized_non_padding_mask.append(np.max(word_piece_masks))

        assert len(detokenized_non_padding_mask) == len(original_untokenized_tokens), \
            f"len(detokenized_non_padding_mask): {len(detokenized_non_padding_mask)}, " \
            f"len(original_untokenized_tokens): {len(original_untokenized_tokens)}, " \
            f"original_untokenized_tokens: {original_untokenized_tokens}"
        non_padding_masks_by_sentence.append(detokenized_non_padding_mask)

    return non_padding_masks_by_sentence

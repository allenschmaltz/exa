import numpy as np

import constants

import codecs
from collections import defaultdict
import torch

import torch.nn.functional as F


def get_sentence_aggregate_filter_values(sentence_to_tokens, token_by_unicnn_filter, expected_filter_size, do_not_apply_relu_on_exemplar_data, main_device):
    # TODO: staged for removal -- this is no longer used by the release code
    sentence_aggregate_filter = torch.zeros(len(sentence_to_tokens), expected_filter_size)
    sentence_aggregate_filter_len_norm = torch.zeros(len(sentence_to_tokens), expected_filter_size)
    for sent_id in sentence_to_tokens:  # this loop over the dictionary is ok here, since sent_id is used as index
        aggregate_filter = torch.zeros(expected_filter_size)
        for token_id in sentence_to_tokens[sent_id]:
            if not do_not_apply_relu_on_exemplar_data:
                aggregate_filter += F.relu(torch.FloatTensor(token_by_unicnn_filter[token_id]))
            else:
                aggregate_filter += torch.FloatTensor(token_by_unicnn_filter[token_id])
        sentence_aggregate_filter[sent_id,:] = aggregate_filter
        sentence_aggregate_filter_len_norm[sent_id,:] = aggregate_filter/len(sentence_to_tokens[sent_id])
    return sentence_aggregate_filter.to(main_device), sentence_aggregate_filter_len_norm.to(main_device)


def get_exemplar_data_from_file(filepath_with_name, expected_filter_size):
    # Note that mask_sent_true_class_0, mask_sent_true_class_1 correspond to true sentence-level labels, so they can be
    # used by the database but typically not for the query
    sentence_to_stats = {}  # sent id -> true sentence-level label, sentence neg logit, sentence pos logit, neg logit bias, pos logit bias
    token_to_sentence = {}  # token id (i.e., row in token_by_unicnn_filter) -> corresponding sentence id
    token_to_sent_relative_token_id = {}  # token id -> corresponding index into the corresponding sentence
    sentence_to_tokens = defaultdict(list)  # sent id -> list of token ids
    token_to_stats = {}  # token id -> true token label, token contribution
    token_by_unicnn_filter = []  # list of filter values
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
                unicnn_filter = tokens[2:]
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


def get_original_sentences(filepath_with_name):

    labels = []
    sentences = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            labels.append(label)
            sentence = line[1:]
            sentences.append(sentence)
    return sentences, labels



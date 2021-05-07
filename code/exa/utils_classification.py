# This includes an evaluation function test() for calculating the accuracy of the sentence-level classifier.
# Additionally, test_based_on_contributions() assesses sentence-level classification using the token-level
# *predictions*.
# Importantly, unlike utils_eval.calculate_seq_metrics(), these methods do not need (nor have access to) ground-truth
# token-level labels. They only have access to the ground-truth *sentence-level* labels.

import utils
import constants
import utils_transformer
import utils_eval
import utils_viz
import utils_exemplar
import utils_sequence_labeling

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


def test(data, model, params, bert_model, bert_device, mode="test"):
    """
    Calculate the sentence-level accuracy and return the output logits
    :param data:
    :param model:
    :param params:
    :param bert_model:
    :param bert_device:
    :param mode:
    :return:
    """
    model.eval()

    if mode == "dev":
        if bert_model is None:
            x, y = data["idx_dev_x"], data["dev_y"]
        else:
            x, bert_idx_sentences, bert_input_masks, y = \
                data["idx_dev_x"], data["dev_bert_idx_sentences"], data["dev_bert_input_masks"], data["dev_y"]
    elif mode == "test":
        if bert_model is None:
            x, y = data["idx_test_x"], data["test_y"]
        else:
            x, bert_idx_sentences, bert_input_masks, y = \
                data["idx_test_x"], data["test_bert_idx_sentences"], data["test_bert_input_masks"], data["test_y"]

    score_vals = []
    pred = []
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)

        batch_x = x[i:i + batch_range]

        if bert_model is not None:
            # get BERT representations
            bert_output = utils_transformer.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                                     bert_input_masks[i:i + batch_range],
                                                                     bert_model, bert_device, params["bert_layers"],
                                                                     len(batch_x[0]))
        else:
            bert_output = None

        batch_y = y[i:i + batch_range]

        if params["GPU"] != -1:
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            if bert_model is not None:
                bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
        else:
            batch_x = Variable(torch.LongTensor(batch_x))
            if bert_model is not None:
                bert_output = Variable(torch.FloatTensor(bert_output))

        model_output = model(batch_x, bert_output,
                             forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION).cpu().data.numpy()

        for j, gold in enumerate(batch_y):
            score_vals.append(f"{gold}\t{model_output[j][0]}\t{model_output[j][1]}\n")
        pred.extend(np.argmax(model_output, axis=1))

    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    if mode == "test":
        # random eval (as a check)
        pred_random = np.random.choice(2, len(pred))
        acc_random = sum([1 if p == y else 0 for p, y in zip(pred_random, y)]) / len(pred_random)
        print(f"\t(Accuracy from random prediction (only for debugging purposes): {acc_random})")
        # always predict 1
        pred_all_ones = np.ones(len(pred))
        acc_all_ones = sum([1 if p == y else 0 for p, y in zip(pred_all_ones, y)]) / len(pred_all_ones)
        print(f"\t(Accuracy from all 1's prediction (only for debugging purposes): {acc_all_ones})")
        print(f"\tGround-truth Stats: Number of instances with class 1: {np.sum(y)} out of {len(y)}")
    return acc, score_vals


def test_based_on_contributions(data, model, params, bert_model, bert_device, mode="test"):
    """
    Calculate the sentence-level accuracy based on the token-level (local) logits and return the min/max output logits.
    Note that y here only contains sentence-level labels, so this can be used in the zero-shot sequence
    labeling setting.
    :param data:
    :param model:
    :param params:
    :param bert_model:
    :param bert_device:
    :param mode:
    :return:
    """
    model.eval()

    if mode == "dev":
        if bert_model is None:
            assert False, f"ERROR: Currently this expects the BERT model and BERT WordPiece tokenization"
            #x, y = data["idx_dev_x"], data["dev_y"]
        else:
            x, bert_idx_sentences, bert_input_masks, y, padded_seq_y_mask = \
                data["idx_dev_x"], data["dev_bert_idx_sentences"], data["dev_bert_input_masks"], \
                data["dev_y"], data["dev_padded_seq_y_mask"]
    elif mode == "test":
        if bert_model is None:
            assert False, f"ERROR: Currently this expects the BERT model and BERT WordPiece tokenization"
            #x, y = data["idx_test_x"], data["test_y"]
        else:
            x, bert_idx_sentences, bert_input_masks, y, padded_seq_y_mask = \
                data["idx_test_x"], data["test_bert_idx_sentences"], data["test_bert_input_masks"], \
                data["test_y"], data["test_padded_seq_y_mask"]

    score_vals = []
    pred = []
    pred_global = []
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)

        batch_x = x[i:i + batch_range]

        if bert_model is not None:
            # get BERT representations
            bert_output = utils_transformer.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                                     bert_input_masks[i:i + batch_range],
                                                                     bert_model, bert_device, params["bert_layers"],
                                                                     len(batch_x[0]))
        else:
            bert_output = None

        batch_y = y[i:i + batch_range]

        if params["GPU"] != -1:
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            if bert_model is not None:
                bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
        else:
            batch_x = Variable(torch.LongTensor(batch_x))
            if bert_model is not None:
                bert_output = Variable(torch.FloatTensor(bert_output))
        batch_padded_seq_y_mask = torch.FloatTensor(padded_seq_y_mask[i:i + batch_range]).to(params["main_device"])

        pred_seq_labels, model_output = \
            model(batch_x, bert_output,
                  forward_type=constants.FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION,
                  main_device=params["main_device"])

        # # Note: The commented line was the original approach, without masking the 0's (i.e., this is a de-facto ReLU)
        # contributions_max, _ = torch.max(batch_padded_seq_y_mask * pred_seq_labels, dim=1)

        contributions_min, _ = torch.min(((1 - batch_padded_seq_y_mask) * (10 ** 8)) + pred_seq_labels, dim=1)
        contributions_max, _ = torch.max(((1 - batch_padded_seq_y_mask) * (-10 ** 8)) + pred_seq_labels, dim=1)

        for j, gold in enumerate(batch_y):
            if contributions_max[j].item() > 0:
                pred.append(1)
            else:
                pred.append(0)
            score_vals.append(f"{gold}\t{contributions_min[j].item()}\t{contributions_max[j].item()}\n")
        pred_global.extend(np.argmax(model_output.cpu().data.numpy(), axis=1))

    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    acc_global = sum([1 if p == y else 0 for p, y in zip(pred_global, y)]) / len(pred_global)
    print(f"\t(Accuracy from the global fc (primarily for debugging purposes): {acc_global})")
    return acc, score_vals

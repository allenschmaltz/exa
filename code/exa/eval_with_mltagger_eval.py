# -*- coding: utf-8 -*-
"""
This was just used to check that our internal evaluation scripts produced the same results as those of the
scripts from the previous work of Rei et al. Note that MLTEvaluator is from the repo associated with those papers.

"""

import sys
import argparse

import string
import codecs

from os import path
import random
from collections import defaultdict
import operator

import numpy as np

from evaluator import MLTEvaluator

import torch

random.seed(1776)


ID_CORRECT = 0 # "negative" class
ID_WRONG = 1  # "positive class"


def get_reference_lines(filepath_with_name):
    true_labels = []
    sentence_labels = []  # token, i|c
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                true_labels.append(sentence_labels)
                sentence_labels = []
            else:
                sentence_labels.append(line)
    assert len(sentence_labels) == 0
    return true_labels


def get_token_and_sentence_scores(filepath_with_name, sentence_level_offset, token_level_offset):
    sentence_scores = []
    token_scores_per_sentence = []
    sentence_score = None
    token_scores = []

    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                sentence_scores.append(sentence_score)
                sentence_score = None
                token_scores_per_sentence.append(token_scores)
                token_scores = []
            else:

                token_score_neg = float(line[0]) + token_level_offset
                token_score_pos = float(line[1])
                neg_sentence_prob = float(line[2])
                pos_sentence_prob = float(line[3])

                prob = torch.nn.functional.softmax(torch.tensor([token_score_neg, token_score_pos]), dim=0)
                token_score_pos = prob[1]

                if sentence_score is None:
                    sentence_score = pos_sentence_prob + sentence_level_offset
                token_scores.append(token_score_pos)
    assert sentence_score is None and token_scores == []
    return sentence_scores, token_scores_per_sentence

def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_generated_detection_file', help="output_generated_detection_file")
    parser.add_argument('--reference_file', help="reference_file")
    parser.add_argument('--sentence_level_offset', type=float, default=0.0, help="sentence_level_offset (if applicable)")
    parser.add_argument('--token_level_offset', type=float, default=0.0,
                        help="token_level_offset (if applicable)")

    args = parser.parse_args(arguments)

    output_generated_detection_file = args.output_generated_detection_file
    reference_file = args.reference_file
    sentence_level_offset = args.sentence_level_offset
    token_level_offset = args.token_level_offset

    reference_labels = get_reference_lines(reference_file)
    sentence_scores, token_scores_per_sentence = get_token_and_sentence_scores(output_generated_detection_file,
                                                                               sentence_level_offset,
                                                                               token_level_offset)
    config = {}
    config["default_label"] = "c"
    evaluator = MLTEvaluator(config)

    cost = -1.0 # not used

    evaluator.append_data(cost, reference_labels, sentence_scores, [np.array(token_scores_per_sentence)])

    results = evaluator.get_results("test")
    for key in results:
        sys.stderr.write(key + ": " + str(results[key]) + "\n")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


# -*- coding: utf-8 -*-
"""
This takes as input binaryevalformat sentences and creates a monolabels file that can be used as a placeholder when
token-level labels are not available or can't be seen for a given experiment.

"""

import sys
import argparse

import string
import codecs

from os import path
from collections import defaultdict
import operator

import numpy as np


def get_lines_and_labels(input_binaryevalformat_file):

    sentences_tokens = []
    sentence_level_labels = []
    token_level_labels = []

    line_id = 0
    with codecs.open(input_binaryevalformat_file, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing {input_binaryevalformat_file} line: {line_id}")
            line = line.strip().split()
            label = int(line[0])
            sentence = line[1:]
            labels = [label] * len(sentence)
            sentences_tokens.append(sentence)
            sentence_level_labels.append(label)
            token_level_labels.append(labels)
            line_id += 1

    return sentences_tokens, sentence_level_labels, token_level_labels


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_binaryevalformat_file', help="input_binaryevalformat_file")
    parser.add_argument('--output_binaryevalformat_labels_file', help="output_binaryevalformat_labels_file")

    args = parser.parse_args(arguments)

    assert args.input_binaryevalformat_file.strip() != args.output_binaryevalformat_labels_file

    sentences_tokens, sentence_level_labels, token_level_labels = \
        get_lines_and_labels(args.input_binaryevalformat_file)

    assert len(sentences_tokens) == len(sentence_level_labels) and len(sentences_tokens) == len(token_level_labels)

    output_binaryevalformat_labels = []

    for tokens, sentence_label, token_labels in zip(sentences_tokens, sentence_level_labels, token_level_labels):
        output_binaryevalformat_labels.append(f"{' '.join([str(x) for x in token_labels])}\n")

    save_lines(args.output_binaryevalformat_labels_file, output_binaryevalformat_labels)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


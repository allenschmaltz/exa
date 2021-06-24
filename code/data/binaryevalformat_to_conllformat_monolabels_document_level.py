# -*- coding: utf-8 -*-
"""
This takes as input binaryevalformat sentences and re-saves as CONLL tab-separated format, where here
we only save monolabels (i.e., the token-level label is always the same as the sentence-level label). This is
similar to binaryevalformat_to_conllformat_monolabels.py, except a separate token-level labels file is not needed.

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

    line_id = 0
    with codecs.open(input_binaryevalformat_file, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing {input_binaryevalformat_file} line: {line_id}")
            line = line.strip().split()
            label = int(line[0])
            sentence = line[1:]

            sentences_tokens.append(sentence)
            sentence_level_labels.append(label)
            line_id += 1

    return sentences_tokens, sentence_level_labels


def convert_to_conll(sentences_tokens, sentence_level_labels):
    conll_output = []
    for sentence, label in zip(sentences_tokens, sentence_level_labels):
        mono_alpha_label = "C" if label == 0 else "U"
        for token in sentence:
            conll_output.append(f"{token}\t{mono_alpha_label}\n")
        conll_output.append("\n")
    return conll_output


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_binaryevalformat_file', help="input_binaryevalformat_file")
    parser.add_argument('--output_conllformat_file', help="output_conllformat_file")

    args = parser.parse_args(arguments)

    sentences_tokens, sentence_level_labels = get_lines_and_labels(args.input_binaryevalformat_file)
    assert len(sentences_tokens) == len(sentence_level_labels)

    conll_output = convert_to_conll(sentences_tokens, sentence_level_labels)
    save_lines(args.output_conllformat_file, conll_output)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


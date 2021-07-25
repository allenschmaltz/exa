# -*- coding: utf-8 -*-
"""
This script takes the paired data and creates sequence labels. We assume that every two lines are paired.

Convert the counterfactual sentiment data (https://github.com/dkaushik96/counterfactually-augmented-data)
 to the binary prediction data format:

0 :: negative sentiment
1 :: positive sentiment

The input is the binary prediction formatted data, which has already been filtered with the BERT tokenizer.

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
import csv

from difflib import SequenceMatcher

random.seed(1776)

CLASS_0_LABEL= "Negative"
CLASS_1_LABEL= "Positive"


INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"



def create_syms(source, target):
    filtered_sent = []
    s = SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace":
            filtered_sent.append(DEL_START_SYM)
            filtered_sent.extend(source[i1:i2])
            filtered_sent.append(DEL_END_SYM)
            filtered_sent.append(INS_START_SYM)
            filtered_sent.extend(target[j1:j2])
            filtered_sent.append(INS_END_SYM)
        elif tag == "delete":
            filtered_sent.append(DEL_START_SYM)
            filtered_sent.extend(source[i1:i2])
            filtered_sent.append(DEL_END_SYM)
        elif tag == "insert":
            filtered_sent.append(INS_START_SYM)
            filtered_sent.extend(target[j1:j2])
            filtered_sent.append(INS_END_SYM)
        elif tag == "equal":
            filtered_sent.extend(target[j1:j2])
        else:
            assert False
    return " ".join(filtered_sent).strip()

def create_positive_sequence_labels(source, target):
    """
    Label 1 for any token covered in a diff to go from positive -> negative, where positive is the *source*
    :param source:
    :param target:
    :return:
    """
    sequence_labels = []
    s = SequenceMatcher(None, source, target, autojunk=False)
    additional_insertions = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace" or tag == "delete":
            tokens = source[i1:i2]
            labels = ["1" for _ in tokens]
            sequence_labels.extend(labels)
        elif tag == "insert":  # re-added later since they need to occur on the next source token
            assert i1 == i2
            additional_insertions.append(i1)
        elif tag == "equal":
            assert source[i1:i2] == target[j1:j2]
            tokens = source[i1:i2]
            labels = ["0" for _ in tokens]
            sequence_labels.extend(labels)
        else:
            assert False
    # flip any insertions not covered
    for insert_index in additional_insertions:
        # here, we make the simplification that any final insertions are just assigned to the last token in the source;
        # this avoids creating a separate holder symbol (and is unlikely to make a difference in practice)
        insert_index = min(insert_index, len(source)-1)
        sequence_labels[insert_index] = "1"
    assert len(sequence_labels) == len(source), f"len(sequence_labels): {len(sequence_labels)}, len(source): {len(source)}"
    return " ".join(sequence_labels).strip()

def get_paired_lines(filepath_with_name):
    """

    """

    seq_labels = []

    current_label_pair = []
    current_sentence_pair = []
    current_positive_index = None
    current_negative_index = None
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            assert label in [0,1]
            sentence = line[1:]
            if len(current_label_pair) == 0:
                current_label_pair.append(label)
                current_sentence_pair.append(sentence)
                if label == 1:
                    current_positive_index = 0
                    current_negative_index = 1
            elif len(current_label_pair) == 1:
                assert current_label_pair[0] != label
                current_label_pair.append(label)
                current_sentence_pair.append(sentence)
                if label == 1:
                    current_positive_index = 1
                    current_negative_index = 0
                # ## temp
                # current_positive_index = 1
                # current_negative_index = 0
                # current_sentence_pair = ["this treatment is not as solid as most".split(),
                #                          "this treatment is as solid as most".split()]

                # process diffs (for reference)
                # diff_from_positive_to_negative = create_syms(current_sentence_pair[current_positive_index], current_sentence_pair[current_negative_index])
                # print(f"{diff_from_positive_to_negative}")

                diff_from_positive_to_negative_sequence_labels = create_positive_sequence_labels(current_sentence_pair[current_positive_index],
                                                             current_sentence_pair[current_negative_index])
                # print(f"{diff_from_positive_to_negative_sequence_labels}")
                # combined_line = []
                # for t, d in zip(current_sentence_pair[current_positive_index], diff_from_positive_to_negative_sequence_labels.split()):
                #     combined_line.append(f"{t}[{d}]")
                # print(f"{' '.join(combined_line)}")
                # save lines
                # we save the sequence labels for two sentences, which must be in the same order as the original
                seq_labels_for_negative_sentence = ["0" for _ in current_sentence_pair[current_negative_index]]
                seq_labels_for_negative_sentence = " ".join(seq_labels_for_negative_sentence).strip()
                if current_negative_index == 0:
                    seq_labels.append(f"{seq_labels_for_negative_sentence}\n")
                    seq_labels.append(f"{diff_from_positive_to_negative_sequence_labels}\n")
                else:
                    seq_labels.append(f"{diff_from_positive_to_negative_sequence_labels}\n")
                    seq_labels.append(f"{seq_labels_for_negative_sentence}\n")

                # update structures
                current_label_pair = []  # not currently used
                current_sentence_pair = []
                current_positive_index = None
                current_negative_index = None

                #exit()
            # if len(sentences) > 3:
            #     exit()
    return seq_labels

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_paired_binaryevalformat_file', type=str, help="input_paired_binaryevalformat_file")
    parser.add_argument('--output_sequence_labels_file', type=str, help="sequence labels (positive sentiment is 1)")

    args = parser.parse_args(arguments)

    input_paired_binaryevalformat_file = args.input_paired_binaryevalformat_file.strip()
    output_sequence_labels_file = args.output_sequence_labels_file.strip()

    assert input_paired_binaryevalformat_file != output_sequence_labels_file

    sequence_labels_lines = get_paired_lines(input_paired_binaryevalformat_file)

    save_lines(output_sequence_labels_file, sequence_labels_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


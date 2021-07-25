# -*- coding: utf-8 -*-
"""
This script takes the paired data and creates output binaryevalformat files and sequence labels to enable
experiments to predict diffs. We assume that every two input lines are paired.

The labels in this case differentiate orig vs. new and NOT sentiment:

0 :: original data
1 :: new data

This means that for the sequence labels (only used for reference) new sentences will have 1's to indicate diffs
to go the original data, whereas original sentences will always have 0's.

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

def create_sequence_labels(source, target):
    """
    Label 1 for any token covered in a diff to go from new -> orig, where new is the *source*
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

def get_lines(filepath_with_name):
    """

    """
    lines = []
    labels = []  # 0 negative, 1 positive
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            assert label in [0,1]
            sentence = " ".join(line[1:])
            lines.append(sentence)
            labels.append(label)
    return lines, labels

def get_paired_lines(filepath_with_name, orig_lines, new_lines):
    """

    """

    seq_labels = []
    sentences_with_domain_labels = []

    current_label_pair = []  # note that this contains sentiment labels, not domain labels
    current_sentence_pair = []
    current_orig_index = None
    current_new_index = None
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            assert label in [0,1]
            sentence = line[1:]
            if len(current_label_pair) == 0:
                current_label_pair.append(label)
                current_sentence_pair.append(sentence)
                if " ".join(sentence) in new_lines:  # the non-verbatim matches in train, dev, test were orig lines, so this should handle that
                    current_new_index = 0
                    current_orig_index = 1
            elif len(current_label_pair) == 1:
                assert current_label_pair[0] != label
                current_label_pair.append(label)
                current_sentence_pair.append(sentence)
                if " ".join(sentence) in new_lines:  # the non-verbatim matches in train, dev, test were orig lines, so this should handle that
                    current_new_index = 1
                    current_orig_index = 0
                # ## temp
                # current_positive_index = 1
                # current_negative_index = 0
                # current_sentence_pair = ["this treatment is not as solid as most".split(),
                #                          "this treatment is as solid as most".split()]

                # process diffs (for reference)
                # diff_from_positive_to_negative = create_syms(current_sentence_pair[current_positive_index], current_sentence_pair[current_negative_index])
                # print(f"{diff_from_positive_to_negative}")

                diff_from_new_to_orig_sequence_labels = create_sequence_labels(current_sentence_pair[current_new_index],
                                                             current_sentence_pair[current_orig_index])
                # print(f"{diff_from_positive_to_negative_sequence_labels}")
                # combined_line = []
                # for t, d in zip(current_sentence_pair[current_positive_index], diff_from_positive_to_negative_sequence_labels.split()):
                #     combined_line.append(f"{t}[{d}]")
                # print(f"{' '.join(combined_line)}")
                # save lines
                # we save the sequence labels for two sentences, which must be in the same order as the original
                seq_labels_for_orig_sentence = ["0" for _ in current_sentence_pair[current_orig_index]]
                seq_labels_for_orig_sentence = " ".join(seq_labels_for_orig_sentence).strip()
                if current_new_index == 0:
                    seq_labels.append(f"{diff_from_new_to_orig_sequence_labels}\n")
                    seq_labels.append(f"{seq_labels_for_orig_sentence}\n")
                    sentences_with_domain_labels.append(f"{1} {' '.join(current_sentence_pair[current_new_index])}\n")
                    sentences_with_domain_labels.append(f"{0} {' '.join(current_sentence_pair[current_orig_index])}\n")
                else:
                    seq_labels.append(f"{seq_labels_for_orig_sentence}\n")
                    seq_labels.append(f"{diff_from_new_to_orig_sequence_labels}\n")
                    sentences_with_domain_labels.append(f"{0} {' '.join(current_sentence_pair[current_orig_index])}\n")
                    sentences_with_domain_labels.append(f"{1} {' '.join(current_sentence_pair[current_new_index])}\n")

                # update structures
                current_label_pair = []  # not currently used
                current_sentence_pair = []
                current_orig_index = None
                current_new_index = None

                #exit()
            # if len(sentences) > 3:
            #     exit()
    assert len(seq_labels) == len(orig_lines) + len(new_lines)
    assert len(seq_labels) == len(sentences_with_domain_labels)
    return seq_labels, sentences_with_domain_labels

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_paired_binaryevalformat_file', type=str, help="input_paired_binaryevalformat_file")
    parser.add_argument('--input_orig_binaryevalformat_file', type=str, help="input_orig_binaryevalformat_file")
    parser.add_argument('--input_new_binaryevalformat_file', type=str, help="input_new_binaryevalformat_file")

    parser.add_argument('--output_paired_domain_binaryevalformat_file', type=str, help="sentence lines (new data is 1)")
    parser.add_argument('--output_paired_domain_sequence_labels_file', type=str, help="sequence labels (new data is 1)")

    parser.add_argument('--output_paired_domain_binaryevalformat_only_orig_file', type=str, help="sentence lines (only orig data)")
    parser.add_argument('--output_paired_domain_sequence_labels_only_orig_file', type=str, help="sequence labels (only orig data)")
    parser.add_argument('--output_paired_domain_binaryevalformat_only_new_file', type=str, help="sentence lines (only new data)")
    parser.add_argument('--output_paired_domain_sequence_labels_only_new_file', type=str, help="sequence labels (only new data)")

    args = parser.parse_args(arguments)

    assert args.input_paired_binaryevalformat_file != args.output_paired_domain_binaryevalformat_file

    orig_lines, orig_labels = get_lines(args.input_orig_binaryevalformat_file) # note that orig_lines does NOT include labels
    new_lines, new_labels = get_lines(args.input_new_binaryevalformat_file)

    assert len(orig_lines) == len(new_lines)

    sequence_labels_lines, sentences_with_domain_labels = get_paired_lines(args.input_paired_binaryevalformat_file, orig_lines, new_lines)

    sequence_labels_lines_only_orig = []
    sentences_with_domain_labels_only_orig = []
    sequence_labels_lines_only_new = []
    sentences_with_domain_labels_only_new = []

    for seq, text in zip(sequence_labels_lines, sentences_with_domain_labels):
        label = int(text[0])
        assert label in [0,1]
        if label == 0:
            sequence_labels_lines_only_orig.append(seq)
            sentences_with_domain_labels_only_orig.append(text)
        elif label == 1:
            sequence_labels_lines_only_new.append(seq)
            sentences_with_domain_labels_only_new.append(text)

    # note that these are aligned with the original combined paired data, from which we can derived the
    # negative/positive sentiment label, if necessary
    save_lines(args.output_paired_domain_sequence_labels_file, sequence_labels_lines)
    save_lines(args.output_paired_domain_binaryevalformat_file, sentences_with_domain_labels)

    # also, need to save versions separated by class (used for analysis, separating true positive and false positive features, etc.)
    save_lines(args.output_paired_domain_binaryevalformat_only_orig_file, sentences_with_domain_labels_only_orig)
    save_lines(args.output_paired_domain_sequence_labels_only_orig_file, sequence_labels_lines_only_orig)
    save_lines(args.output_paired_domain_binaryevalformat_only_new_file, sentences_with_domain_labels_only_new)
    save_lines(args.output_paired_domain_sequence_labels_only_new_file, sequence_labels_lines_only_new)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


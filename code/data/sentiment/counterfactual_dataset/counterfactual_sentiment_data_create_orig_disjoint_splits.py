# -*- coding: utf-8 -*-
"""
Here we take the paired data of (https://github.com/dkaushik96/counterfactually-augmented-data) and the full original
data and we create splits in which we exclude the original review corresponding to any revised review. The result is
that we can assess whether having both the source and target is driving the effectiveness difference, or if it is
largely due to domain differences (i.e., unique language used in the revised vs. original reviews).

For the train split, we create two datasets:

1. 1.7k revised + (19k original - the 1.7k original sentences that correspond to the revised data)
2. 1.7k revised + (1.7k original sentences that do not correspond to the revised data)

For the dev split, for each pair in the dev paired file, we randomly select one of original or revised. The resulting
dev set is half the size of the paired file.

The output is the standard binary prediction data format:

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

from sklearn.utils import shuffle

random.seed(1776)

CLASS_0_LABEL= "Negative"
CLASS_1_LABEL= "Positive"

def get_disjoint_lines_from_paired_lines(filepath_with_name, np_random_state):
    """
    This assumes that every two lines in filepath_with_name are paired.
    """

    disjoint_lines = []

    current_label_pair = []
    current_line_pair = []
    total_neg = 0
    total_pos = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = f"{line.strip()}\n"
            label = int(line.split()[0])
            assert label in [0,1]
            if len(current_label_pair) == 0:
                current_label_pair.append(label)
                current_line_pair.append(line)

            elif len(current_label_pair) == 1:
                assert current_label_pair[0] != label
                current_label_pair.append(label)
                current_line_pair.append(line)

                current_label_pair, current_line_pair = shuffle(current_label_pair, current_line_pair, random_state=np_random_state)
                disjoint_lines.append(current_line_pair[0]) # always take the first index of shuffled
                if current_label_pair[0] == 0:
                    total_neg += 1
                else:
                    total_pos += 1
                current_label_pair = []
                current_line_pair = []
    print(f"In the {len(disjoint_lines)} disjoint lines there are: Negative lines: {total_neg}; Positive lines: {total_pos}")
    return disjoint_lines


def get_lines(filepath_with_name):
    """

    """
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(f"{line}\n")
    return lines


def get_lines_filtered(filepath_with_name, filter_lines):
    """

    """
    lines = []
    num_filtered = 0
    filter_lines_dict = defaultdict(int)
    for line in filter_lines:
        filter_lines_dict[line] += 1
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = f"{line.strip()}\n"
            if line not in filter_lines_dict:
                lines.append(line)
            else:
                num_filtered += 1
                filter_lines_dict[line] = -1

    print(f"Number of filtered lines: {num_filtered}")
    remaining_filtered_lines = []
    for line in filter_lines_dict:
        if filter_lines_dict[line] > 0:
            assert line not in lines
            remaining_filtered_lines.append(line)

    return lines, remaining_filtered_lines

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--input_train_combined_paired_binaryevalformat_file', type=str, help="input_train_combined_paired_binaryevalformat_file")
    parser.add_argument('--input_train_full_orig_binaryevalformat_file', type=str, help="input_train_full_orig_binaryevalformat_file")
    parser.add_argument('--final_sample_size', type=int, default=3414, help="final_sample_size")

    parser.add_argument('--input_dev_combined_paired_binaryevalformat_file', type=str,
                        help="input_dev_combined_paired_binaryevalformat_file")

    parser.add_argument('--output_train_disjoint_3k_binaryevalformat_file', type=str, help="output_train_disjoint_3k_binaryevalformat_file")
    parser.add_argument('--output_train_disjoint_full_binaryevalformat_file', type=str,
                        help="output_train_disjoint_full_binaryevalformat_file")

    parser.add_argument('--output_dev_disjoint_binaryevalformat_file', type=str, help="output_dev_disjoint_binaryevalformat_file")

    args = parser.parse_args(arguments)

    assert args.input_train_combined_paired_binaryevalformat_file not in [args.output_train_disjoint_3k_binaryevalformat_file, args.output_train_disjoint_full_binaryevalformat_file]
    assert args.input_dev_combined_paired_binaryevalformat_file != args.output_dev_disjoint_binaryevalformat_file


    orig_lines = get_lines(args.input_train_full_orig_binaryevalformat_file)
    disjoint_paired_lines, remaining_full_orig_lines = get_lines_filtered(args.input_train_combined_paired_binaryevalformat_file, orig_lines)
    print(f"Length of disjoint_paired_lines: {len(disjoint_paired_lines)}; length of remaining_full_orig_lines: {len(remaining_full_orig_lines)}")

    train_disjoint_full = list(disjoint_paired_lines)
    train_disjoint_full.extend(list(remaining_full_orig_lines))
    print(f"Length of train_disjoint_full: {len(train_disjoint_full)}")
    save_lines(args.output_train_disjoint_full_binaryevalformat_file, train_disjoint_full)

    remaining_sample_size = args.final_sample_size - len(disjoint_paired_lines)
    np_random_state = np.random.RandomState(1776)
    remaining_full_orig_lines = shuffle(remaining_full_orig_lines, random_state = np_random_state)

    train_disjoint_3k = list(disjoint_paired_lines)
    train_disjoint_3k.extend(remaining_full_orig_lines[0:remaining_sample_size])

    assert len(train_disjoint_3k) == args.final_sample_size
    save_lines(args.output_train_disjoint_3k_binaryevalformat_file, train_disjoint_3k)

    # dev set
    dev_disjoint_lines = get_disjoint_lines_from_paired_lines(args.input_dev_combined_paired_binaryevalformat_file, np_random_state)
    save_lines(args.output_dev_disjoint_binaryevalformat_file, dev_disjoint_lines)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


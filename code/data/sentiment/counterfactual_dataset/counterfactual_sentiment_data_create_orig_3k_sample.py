# -*- coding: utf-8 -*-
"""
The original repo of (https://github.com/dkaushik96/counterfactually-augmented-data) does not include the 3.4k sample
of original data. We create that here from the original 1.7k sample and a disjoint sample from the full original file.

(This is only for the train split. The dev and test splits are the same as before.)


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
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = f"{line.strip()}\n"
            if line not in filter_lines:
                lines.append(line)
            else:
                num_filtered += 1
    print(f"Number of filtered lines: {num_filtered}")
    return lines

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_orig_binaryevalformat_file', type=str, help="input_orig_binaryevalformat_file")
    parser.add_argument('--input_full_orig_binaryevalformat_file', type=str, help="input_full_orig_binaryevalformat_file")
    parser.add_argument('--final_sample_size', type=int, help="desired size of the output sample"
                                                              "(must be > size of input_orig_binaryevalformat_file "
                                                              "and < size of input_full_orig_binaryevalformat_file)")
    parser.add_argument('--output_binaryevalformat_file', type=str, help="output_binaryevalformat_file")


    args = parser.parse_args(arguments)


    assert args.input_orig_binaryevalformat_file != args.output_binaryevalformat_file
    assert args.input_full_orig_binaryevalformat_file != args.output_binaryevalformat_file

    orig_lines = get_lines(args.input_orig_binaryevalformat_file)
    remaining_full_orig_lines = get_lines_filtered(args.input_full_orig_binaryevalformat_file, orig_lines)

    remaining_sample_size = args.final_sample_size - len(orig_lines)
    np_random_state = np.random.RandomState(1776)
    remaining_full_orig_lines = shuffle(remaining_full_orig_lines, random_state = np_random_state)

    orig_lines.extend(remaining_full_orig_lines[0:remaining_sample_size])

    assert len(orig_lines) == args.final_sample_size
    save_lines(args.output_binaryevalformat_file, orig_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


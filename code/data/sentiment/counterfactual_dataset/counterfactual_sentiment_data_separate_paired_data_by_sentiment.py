# -*- coding: utf-8 -*-
"""
This script takes the paired data and the domain data and separate the domain data by sentiment.

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


random.seed(1776)

CLASS_0_LABEL= "Negative"
CLASS_1_LABEL= "Positive"



def get_lines(filepath_with_name, get_labels=True):
    """

    """
    lines = []
    labels = []  # 0 negative, 1 positive, or 0 orig, 1 new domain
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if get_labels:
                label = int(line[0])
                assert label in [0,1]
                labels.append(label)
            sentence = f"{' '.join(line)}\n"
            lines.append(sentence)
    if get_labels:
        assert len(lines) == len(labels)
    return lines, labels



def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_paired_sentiment_binaryevalformat_file', type=str, help="paired data (where the labels are sentiment)")
    parser.add_argument('--input_paired_domain_binaryevalformat_file', type=str, help="paired data (where the labels are domains)")
    parser.add_argument('--input_paired_domain_sequence_labels_file', type=str, help="paired sequence label data (where the labels are domains)")

    parser.add_argument('--output_paired_domain_binaryevalformat_only_negative_sentiment_file', type=str, help="sentence lines (only negative sentiment data, but labels are domain)")
    parser.add_argument('--output_paired_domain_sequence_labels_only_negative_sentiment_file', type=str, help="sequence labels (only negative sentiment data, but labels are domain)")
    parser.add_argument('--output_paired_domain_binaryevalformat_only_positive_sentiment_file', type=str, help="sentence lines (only positive sentiment data, but labels are domain)")
    parser.add_argument('--output_paired_domain_sequence_labels_only_positive_sentiment_file', type=str, help="sequence labels (only positive sentiment data, but labels are domain)")

    parser.add_argument('--output_paired_domain_binaryevalformat_only_negative_sentiment_only_orig_file', type=str, help="sentence lines (only negative sentiment data, but labels are domain) only orig domain")
    parser.add_argument('--output_paired_domain_sequence_labels_only_negative_sentiment_only_orig_file', type=str, help="sequence labels (only negative sentiment data, but labels are domain) only orig domain")
    parser.add_argument('--output_paired_domain_binaryevalformat_only_positive_sentiment_only_orig_file', type=str, help="sentence lines (only positive sentiment data, but labels are domain) only orig domain")
    parser.add_argument('--output_paired_domain_sequence_labels_only_positive_sentiment_only_orig_file', type=str, help="sequence labels (only positive sentiment data, but labels are domain) only orig domain")

    parser.add_argument('--output_paired_domain_binaryevalformat_only_negative_sentiment_only_new_file', type=str, help="sentence lines (only negative sentiment data, but labels are domain) only new domain")
    parser.add_argument('--output_paired_domain_sequence_labels_only_negative_sentiment_only_new_file', type=str, help="sequence labels (only negative sentiment data, but labels are domain) only new domain")
    parser.add_argument('--output_paired_domain_binaryevalformat_only_positive_sentiment_only_new_file', type=str, help="sentence lines (only positive sentiment data, but labels are domain) only new domain")
    parser.add_argument('--output_paired_domain_sequence_labels_only_positive_sentiment_only_new_file', type=str, help="sequence labels (only positive sentiment data, but labels are domain) only new domain")



    args = parser.parse_args(arguments)

    assert args.input_paired_sentiment_binaryevalformat_file != args.input_paired_domain_binaryevalformat_file

    sentiment_lines, sentiment_labels = get_lines(args.input_paired_sentiment_binaryevalformat_file)

    domain_lines, domain_labels = get_lines(args.input_paired_domain_binaryevalformat_file)
    domain_seq_labels, _ = get_lines(args.input_paired_domain_sequence_labels_file, get_labels=False)

    assert len(sentiment_lines) == len(domain_lines)
    assert len(domain_lines) == len(domain_seq_labels)

    only_neg = []
    only_neg_seq_labels = []
    only_pos = []
    only_pos_seq_labels = []

    only_neg_orig = []
    only_neg_seq_labels_orig = []
    only_pos_orig = []
    only_pos_seq_labels_orig = []

    only_neg_new = []
    only_neg_seq_labels_new = []
    only_pos_new = []
    only_pos_seq_labels_new = []


    for sentiment_line, sentiment_label, domain_line, domain_label, domain_seq_label in zip(sentiment_lines, sentiment_labels, domain_lines, domain_labels, domain_seq_labels):
        assert sentiment_line[1:] == domain_line[1:]
        if sentiment_label == 0:
            only_neg.append(domain_line)
            only_neg_seq_labels.append(domain_seq_label)
            if domain_label == 0:
                only_neg_orig.append(domain_line)
                only_neg_seq_labels_orig.append(domain_seq_label)
            elif domain_label == 1:
                only_neg_new.append(domain_line)
                only_neg_seq_labels_new.append(domain_seq_label)
        elif sentiment_label == 1:
            only_pos.append(domain_line)
            only_pos_seq_labels.append(domain_seq_label)
            if domain_label == 0:
                only_pos_orig.append(domain_line)
                only_pos_seq_labels_orig.append(domain_seq_label)
            elif domain_label == 1:
                only_pos_new.append(domain_line)
                only_pos_seq_labels_new.append(domain_seq_label)

    save_lines(args.output_paired_domain_binaryevalformat_only_negative_sentiment_file, only_neg)
    save_lines(args.output_paired_domain_sequence_labels_only_negative_sentiment_file, only_neg_seq_labels)

    save_lines(args.output_paired_domain_binaryevalformat_only_positive_sentiment_file, only_pos)
    save_lines(args.output_paired_domain_sequence_labels_only_positive_sentiment_file, only_pos_seq_labels)

    save_lines(args.output_paired_domain_binaryevalformat_only_negative_sentiment_only_orig_file, only_neg_orig)
    save_lines(args.output_paired_domain_sequence_labels_only_negative_sentiment_only_orig_file, only_neg_seq_labels_orig)
    save_lines(args.output_paired_domain_binaryevalformat_only_positive_sentiment_only_orig_file, only_pos_orig)
    save_lines(args.output_paired_domain_sequence_labels_only_positive_sentiment_only_orig_file, only_pos_seq_labels_orig)

    save_lines(args.output_paired_domain_binaryevalformat_only_negative_sentiment_only_new_file, only_neg_new)
    save_lines(args.output_paired_domain_sequence_labels_only_negative_sentiment_only_new_file, only_neg_seq_labels_new)
    save_lines(args.output_paired_domain_binaryevalformat_only_positive_sentiment_only_new_file, only_pos_new)
    save_lines(args.output_paired_domain_sequence_labels_only_positive_sentiment_only_new_file, only_pos_seq_labels_new)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


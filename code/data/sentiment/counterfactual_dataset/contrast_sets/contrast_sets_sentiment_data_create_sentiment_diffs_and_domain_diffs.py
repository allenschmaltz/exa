# -*- coding: utf-8 -*-
"""
This script takes the original and contrast data and creates output binaryevalformat files and sequence labels to enable
experiments to predict token-level sentiment diffs, review-level domain (original vs. revised), and token-level
domain diffs.
We assume that the original and contrast files are parallel.

The labels in this case differentiate orig vs. new as follows:

0 :: original data
1 :: new data

This means that for the sequence labels (only used for reference) new sentences will have 1's to indicate diffs
to go to the original data, whereas original sentences will always have 0's.

Similarly for sentiment:

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

def create_sequence_labels(source, target):
    """
    For domain diffs: Label 1 for any token covered in a diff to go from new -> orig, where new is the *source*
    For sentiment diffs: Label 1 for any token covered in a diff to go from positive -> negative, where positive is the *source*
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

def get_and_save_paired_lines(orig_lines, orig_labels, new_lines, new_labels, output_sentiment_sequence_file_prefix, output_domain_file_prefix):
    """
    Save lines, domain diff, domain labels, sentiment diff, sentiment labels and subsets

    We have original lines and revised lines, and their associated *sentiment* labels, and we want to create subsets
    based on sentiment and domain (original or revised), including token-level diffs (for sentiment, and separately,
    for domain). To ensure continuity of the analyses, it is important to not conflate the sentiment labels (0 is negative
    and 1 is positive) and the domain labels (0 is original, 1 is revised).
    """

    sentiment_seq_labels_orig = []
    sentiment_seq_labels_new = []

    # rows of Review-level Domain (Not Sentiment) table (~Table 9 in the draft)
    domain_seq_labels = []
    sentences_with_domain_labels = []

    domain_seq_labels_only_orig = []
    sentences_with_domain_labels_only_orig = []
    domain_seq_labels_only_new = []
    sentences_with_domain_labels_only_new = []

    domain_seq_labels_only_neg = []
    sentences_with_domain_labels_only_neg = []
    domain_seq_labels_only_pos = []
    sentences_with_domain_labels_only_pos = []

    domain_seq_labels_only_orig_only_neg = []
    sentences_with_domain_labels_only_orig_only_neg = []
    domain_seq_labels_only_orig_only_pos = []
    sentences_with_domain_labels_only_orig_only_pos = []

    domain_seq_labels_only_new_only_neg = []
    sentences_with_domain_labels_only_new_only_neg = []
    domain_seq_labels_only_new_only_pos = []
    sentences_with_domain_labels_only_new_only_pos = []

    for orig_line, orig_sentiment_label, new_line, new_sentiment_label in zip(orig_lines, orig_labels, new_lines, new_labels):
        orig_line = orig_line.split()
        new_line = new_line.split()
        ## sentiment sequence labels:
        if orig_sentiment_label == 0:
            assert new_sentiment_label == 1

            # uncomment this to see the source-target diffs
            # print(create_syms(new_line, orig_line))

            diff_from_positive_to_negative_sequence_labels = create_sequence_labels(new_line, orig_line)
            # annotated = []
            # for token, diff in zip(new_line, diff_from_positive_to_negative_sequence_labels.split()):
            #     annotated.append(f"{token}({diff})")

            sentiment_seq_labels_orig.append(" ".join(["0" for _ in orig_line]).strip()+"\n")
            sentiment_seq_labels_new.append(f"{diff_from_positive_to_negative_sequence_labels}\n")
        else:
            assert new_sentiment_label == 0
            diff_from_positive_to_negative_sequence_labels = create_sequence_labels(orig_line, new_line)
            sentiment_seq_labels_orig.append(f"{diff_from_positive_to_negative_sequence_labels}\n")
            sentiment_seq_labels_new.append(" ".join(["0" for _ in new_line]).strip() + "\n")

        ## domain sequence labels:
        diff_from_new_to_orig_sequence_labels = create_sequence_labels(new_line, orig_line)

        seq_labels_for_orig_sentence = ["0" for _ in orig_line]
        seq_labels_for_orig_sentence = " ".join(seq_labels_for_orig_sentence).strip()

        # for original lines followed by new lines:
        domain_seq_labels.append(f"{seq_labels_for_orig_sentence}\n")
        domain_seq_labels.append(f"{diff_from_new_to_orig_sequence_labels}\n")
        sentences_with_domain_labels.append(f"{0} {' '.join(orig_line)}\n")
        sentences_with_domain_labels.append(f"{1} {' '.join(new_line)}\n")

        # various subsets used for further analyses:
        domain_seq_labels_only_orig.append(f"{seq_labels_for_orig_sentence}\n")
        sentences_with_domain_labels_only_orig.append(f"{0} {' '.join(orig_line)}\n")
        domain_seq_labels_only_new.append(f"{diff_from_new_to_orig_sequence_labels}\n")
        sentences_with_domain_labels_only_new.append(f"{1} {' '.join(new_line)}\n")

        if orig_sentiment_label == 0:
            domain_seq_labels_only_neg.append(f"{seq_labels_for_orig_sentence}\n")
            sentences_with_domain_labels_only_neg.append(f"{0} {' '.join(orig_line)}\n")
            domain_seq_labels_only_pos.append(f"{diff_from_new_to_orig_sequence_labels}\n")
            sentences_with_domain_labels_only_pos.append(f"{1} {' '.join(new_line)}\n")

            domain_seq_labels_only_orig_only_neg.append(f"{seq_labels_for_orig_sentence}\n")
            sentences_with_domain_labels_only_orig_only_neg.append(f"{0} {' '.join(orig_line)}\n")

            domain_seq_labels_only_new_only_pos.append(f"{diff_from_new_to_orig_sequence_labels}\n")
            sentences_with_domain_labels_only_new_only_pos.append(f"{1} {' '.join(new_line)}\n")
        else:
            domain_seq_labels_only_neg.append(f"{diff_from_new_to_orig_sequence_labels}\n")
            sentences_with_domain_labels_only_neg.append(f"{1} {' '.join(new_line)}\n")
            domain_seq_labels_only_pos.append(f"{seq_labels_for_orig_sentence}\n")
            sentences_with_domain_labels_only_pos.append(f"{0} {' '.join(orig_line)}\n")

            domain_seq_labels_only_orig_only_pos.append(f"{seq_labels_for_orig_sentence}\n")
            sentences_with_domain_labels_only_orig_only_pos.append(f"{0} {' '.join(orig_line)}\n")

            domain_seq_labels_only_new_only_neg.append(f"{diff_from_new_to_orig_sequence_labels}\n")
            sentences_with_domain_labels_only_new_only_neg.append(f"{1} {' '.join(new_line)}\n")



    assert len(sentiment_seq_labels_orig) == len(orig_lines)
    assert len(sentiment_seq_labels_new) == len(new_lines)

    assert len(domain_seq_labels) == len(orig_lines) + len(new_lines)
    assert len(domain_seq_labels) == len(sentences_with_domain_labels)

    # sentiment diffs (here, we only save the seq labels since the corresponding text files are unchanged)
    output_sentiment_sequence_file_prefix = output_sentiment_sequence_file_prefix.strip()
    save_lines(f"{output_sentiment_sequence_file_prefix}.sentiment_diffs_sequence_labels.orig.txt", sentiment_seq_labels_orig)
    save_lines(f"{output_sentiment_sequence_file_prefix}.sentiment_diffs_sequence_labels.new.txt", sentiment_seq_labels_new)

    # domain diffs
    output_domain_file_prefix = output_domain_file_prefix.strip()
    suffix_label = "all"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels)


    suffix_label = "only_orig"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_orig)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_orig)

    suffix_label = "only_new"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_new)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_new)

    suffix_label = "only_neg"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_neg)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_neg)

    suffix_label = "only_pos"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_pos)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_pos)

    suffix_label = "only_orig_only_neg"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_orig_only_neg)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_orig_only_neg)

    suffix_label = "only_orig_only_pos"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_orig_only_pos)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_orig_only_pos)

    suffix_label = "only_new_only_neg"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_new_only_neg)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_new_only_neg)

    suffix_label = "only_new_only_pos"
    save_lines(f"{output_domain_file_prefix}.domain_diffs_sequence_labels.{suffix_label}.txt", domain_seq_labels_only_new_only_pos)
    save_lines(f"{output_domain_file_prefix}.domain.{suffix_label}.txt", sentences_with_domain_labels_only_new_only_pos)


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_orig_binaryevalformat_file', type=str, help="input_orig_binaryevalformat_file")
    parser.add_argument('--input_new_binaryevalformat_file', type=str, help="input_new_binaryevalformat_file")

    parser.add_argument('--output_sentiment_sequence_file_prefix', type=str, help="output_sentiment_sequence_file_prefix")
    parser.add_argument('--output_domain_file_prefix', type=str, help="output_domain_file_prefix")


    args = parser.parse_args(arguments)

    orig_lines, orig_labels = get_lines(args.input_orig_binaryevalformat_file) # note that orig_lines does NOT include labels
    new_lines, new_labels = get_lines(args.input_new_binaryevalformat_file)

    assert len(orig_lines) == len(new_lines)

    get_and_save_paired_lines(orig_lines, orig_labels, new_lines, new_labels, args.output_sentiment_sequence_file_prefix,
                     args.output_domain_file_prefix)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


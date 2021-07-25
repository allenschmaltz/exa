# -*- coding: utf-8 -*-
"""
This script takes the paired data and separates new and orig using the unpaired files. Sequence label files are also
separated. (The point of doing this is to align the sequence label diffs. The resulting orig and new files should,
in principle, have the same information as the orig and new files in the original repo, but now the sentences
are aligned with the diffs.)

Note that 3 lines in train are not matched and are dropped, making the orig split 3 lines shorter than the originals.

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

def get_lines(filepath_with_name):
    """

    """
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(f"{line}\n")
    return lines

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_split', type=str, help="train,dev,test")
    parser.add_argument('--input_paired_binaryevalformat_file', type=str, help="input_paired_binaryevalformat_file")
    parser.add_argument('--input_paired_sequence_labels_file', type=str, help="input_paired_sequence_labels_file")
    parser.add_argument('--input_orig_binaryevalformat_file', type=str, help="input_orig_binaryevalformat_file")
    parser.add_argument('--input_new_binaryevalformat_file', type=str, help="input_new_binaryevalformat_file")
    parser.add_argument('--output_orig_binaryevalformat_file', type=str,
                        help="orig")
    parser.add_argument('--output_new_binaryevalformat_file', type=str,
                        help="new")
    parser.add_argument('--output_orig_sequence_labels_file', type=str, help="sequence labels (positive sentiment is 1)")
    parser.add_argument('--output_new_sequence_labels_file', type=str, help="sequence labels (positive sentiment is 1)")
    args = parser.parse_args(arguments)


    assert args.input_paired_binaryevalformat_file != args.output_orig_binaryevalformat_file
    assert args.input_paired_binaryevalformat_file != args.output_new_binaryevalformat_file

    paired_lines = get_lines(args.input_paired_binaryevalformat_file)
    paired_seq_lines = get_lines(args.input_paired_sequence_labels_file)

    orig_lines = get_lines(args.input_orig_binaryevalformat_file)
    new_lines = get_lines(args.input_new_binaryevalformat_file)

    remaining_orig_lines = list(orig_lines)
    remaining_new_lines = list(new_lines)

    output_orig_lines = []
    output_new_lines = []
    output_orig_seq_lines = []
    output_new_seq_lines = []

    remaining_indexes = []  # indexes in the combined data without a verbatim match in orig or new
    for i, (line, seq) in enumerate(zip(paired_lines, paired_seq_lines)):
        if line in orig_lines:
            assert line not in new_lines
            output_orig_lines.append(line)
            output_orig_seq_lines.append(seq)
            line_index = orig_lines.index(line)
            remaining_orig_lines[line_index] = None
        elif line in new_lines:
            assert line not in orig_lines
            output_new_lines.append(line)
            output_new_seq_lines.append(seq)
            line_index = new_lines.index(line)
            remaining_new_lines[line_index] = None
        else:
            print(f"WARNING: The following line (at index {i}) was not verbatim in new or orig:")
            print(line)
            remaining_indexes.append(i)

    print(f"len(remaining_indexes): {len(remaining_indexes)}; {remaining_indexes}")

    possible_orig_lines = []
    possible_new_lines = []

    for line in remaining_orig_lines:
        if line is not None:
            possible_orig_lines.append(line)

    for line in possible_new_lines:
        if line is not None:
            possible_new_lines.append(line)

    print(f"Possible orig lines:")
    print(possible_orig_lines)

    print(f"Possible new lines:")
    print(possible_new_lines)

    if args.input_split == "dev":
        # index 170
        # here, for some reason, the original was missing <br /><br /> but otherwise appears to be the same, so we add manually
        assert remaining_indexes == [170]
        assert len(remaining_indexes) == 1
        assert len(possible_orig_lines) == 1 and len(possible_new_lines) == 0
        output_orig_lines.append(paired_lines[remaining_indexes[0]])
        output_orig_seq_lines.append(paired_seq_lines[remaining_indexes[0]])
    elif args.input_split == "test":
        # index 378
        assert remaining_indexes == [378]
        # for some reason, the original has different quotation escapes
        assert len(remaining_indexes) == 1
        assert len(possible_orig_lines) == 1 and len(possible_new_lines) == 0
        # diff = create_syms(possible_orig_lines[0].split(), paired_lines[remaining_indexes[0]].split())
        # print(f"diff: {diff}")
        output_orig_lines.append(paired_lines[remaining_indexes[0]])
        output_orig_seq_lines.append(paired_seq_lines[remaining_indexes[0]])
    elif args.input_split == "train":
        assert remaining_indexes == [848, 2322, 2660]
        # in train, this last index 2660 does not seem to actually match anything; and 848 flips a negation;
        # since it is the train split and just 3 lines, so we just drop all 3 of these rather than re-align
        # for possible_line, remain_index in zip(possible_orig_lines, remaining_indexes):
        #     diff = create_syms(possible_line.split(), paired_lines[remain_index].split())
        #     print(f"diff: {diff}")

    if args.input_split in ["dev", "test"]:
        assert len(orig_lines) == len(output_orig_lines)
        assert len(orig_lines) == len(output_orig_seq_lines)
        assert len(new_lines) == len(output_new_lines)
        assert len(new_lines) == len(output_new_seq_lines)
    else:
        assert len(orig_lines)-3 == len(output_orig_lines)
        assert len(orig_lines)-3 == len(output_orig_seq_lines)
        assert len(new_lines) == len(output_new_lines)
        assert len(new_lines) == len(output_new_seq_lines)

    save_lines(args.output_orig_binaryevalformat_file, output_orig_lines)
    save_lines(args.output_orig_sequence_labels_file, output_orig_seq_lines)

    save_lines(args.output_new_binaryevalformat_file, output_new_lines)
    save_lines(args.output_new_sequence_labels_file, output_new_seq_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


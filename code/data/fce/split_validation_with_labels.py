# -*- coding: utf-8 -*-
"""

Used to split the validation data into smaller training and validation sets for semi-supervised tagging.

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

random.seed(1776)

UNK_SYM = "unk"

INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"


def get_lines(filepath_with_name):
    lines = []
    idx = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if idx % 10000 == 0:
                print(f"Currently processing line {idx}")

            line = line.strip()
            lines.append(line + "\n")
            idx += 1
    return lines


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--valid_binaryeval_file', default="", help='valid_binaryeval_file')
    parser.add_argument("--valid_labels_file", default="", help="valid_labels_file")
    parser.add_argument('--split_0_size', type=int, default=500,
                        help="size of split 0; remaining sentences will be in split 1")
    parser.add_argument('--output_binaryeval_split_file_template', default="binaryeval_split_num",
                        help="output_binaryeval_split_template")
    parser.add_argument('--output_labels_split_file_template', default="labels_split_num",
                        help="output_labels_split_template")
    parser.add_argument('--output_index_file_template', default="index_split_num",
                        help="Record of indexes in split 0 and split 1")

    args = parser.parse_args(arguments)

    valid_binaryeval_file = args.valid_binaryeval_file
    valid_labels_file = args.valid_labels_file
    split_0_size = args.split_0_size
    output_binaryeval_split_file_template = args.output_binaryeval_split_file_template
    output_labels_split_file_template = args.output_labels_split_file_template
    output_index_file_template = args.output_index_file_template

    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)

    lines = get_lines(valid_binaryeval_file)
    labels = get_lines(valid_labels_file)
    sample_indecies = np_random_state.choice(len(lines), split_0_size, replace=False)

    split0_indecies = {}
    for idx in sample_indecies:
        split0_indecies[idx] = 1

    split0_lines = []
    split0_labels = []
    split0_idx_lines = []

    split1_lines = []
    split1_labels = []
    split1_idx_lines = []

    for idx in range(0, len(lines)):
        if idx in split0_indecies:
            split0_lines.append(lines[idx])
            split0_labels.append(labels[idx])
            split0_idx_lines.append(f"{idx}\n")
        else:
            split1_lines.append(lines[idx])
            split1_labels.append(labels[idx])
            split1_idx_lines.append(f"{idx}\n")

    print(f"Size of split 0: {len(split0_lines)}")
    print(f"Size of split 1: {len(split1_lines)}")

    split_0_file_suffix = f"_split0_size{len(split0_lines)}.txt"
    split_1_file_suffix = f"_split1_size{len(split1_lines)}.txt"

    save_lines(output_binaryeval_split_file_template + split_0_file_suffix, split0_lines)
    save_lines(output_labels_split_file_template + split_0_file_suffix, split0_labels)
    save_lines(output_index_file_template + split_0_file_suffix, split0_idx_lines)

    save_lines(output_binaryeval_split_file_template + split_1_file_suffix, split1_lines)
    save_lines(output_labels_split_file_template + split_1_file_suffix, split1_labels)
    save_lines(output_index_file_template + split_1_file_suffix, split1_idx_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


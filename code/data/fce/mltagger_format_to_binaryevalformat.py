# -*- coding: utf-8 -*-
"""
Convert MLTagger format of Rei 2018 to the binary eval format

Labels have the same length as the lines -- i.e., a final holder symbol has not been added.

The format for --output_identification_file is as follows, where there is a single space between the document-level
label and the tokens of the document:
0|1 Tokens in the document\n

The format for --output_identification_labels_file is a follows, with a 0 or 1 for every token corresponding to the
lines in --output_identification_file:
0|1 for every token in the document\n

"""

import sys
import argparse

import string
import codecs

from os import path
import random
from collections import defaultdict
import operator

random.seed(1776)

ID_CORRECT = 0 # "negative" class (correct token)
ID_WRONG = 1  # "positive class" (token with error)

MLTAGGER_ID_CORRECT = "c"
MLTAGGER_ID_WRONG = "i"


def get_lines(filepath_with_name):

    lines = []
    labels = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        sentence = []
        sentence_labels = []
        class_label = ID_CORRECT
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                token = line[0]
                label = line[1]
                sentence.append(token)
                if label == MLTAGGER_ID_CORRECT:
                    label = ID_CORRECT
                else:
                    label = ID_WRONG
                    class_label = ID_WRONG
                sentence_labels.append(str(label))
            else:
                assert len(line) == 0
                assert len(sentence) == len(sentence_labels) and len(sentence) != 0
                lines.append(" ".join([str(class_label)] + sentence) + "\n")
                labels.append(" ".join(sentence_labels) + "\n")
                sentence = []
                sentence_labels = []
                class_label = ID_CORRECT
    assert len(sentence) == 0 and len(sentence_labels) == 0
    return lines, labels


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_mltagger_format_file', help="input_mltagger_format_file")
    parser.add_argument('--output_identification_file', help="output_identification_file")
    parser.add_argument('--output_identification_labels_file', help="output_identification_labels_file")

    args = parser.parse_args(arguments)

    input_mltagger_format_file = args.input_mltagger_format_file
    output_identification_file = args.output_identification_file
    output_identification_labels_file = args.output_identification_labels_file

    lines, labels = get_lines(input_mltagger_format_file)

    print(f"Number of lines: {len(lines)}")

    save_lines(output_identification_file, lines)
    save_lines(output_identification_labels_file, labels)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


# -*- coding: utf-8 -*-
"""

Split a binary prediction data format file into dev, test, and varying training sizes. Here, it is assumed that
the sentence-level label here is always 0.

Token-level labels are also generated (again, here, always 0).

Any token filtered by the BERT tokenizer is ignored.

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
from sklearn.utils import shuffle

from pytorch_pretrained_bert.tokenization import BertTokenizer

random.seed(1776)


def get_lines(filepath_with_name, tokenizer):
    lines = []  # prefix label + sentence + \n
    token_labels = []  # token labels + \n

    num_sentences_with_filtered_tokens = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            assert label == 0
            sentence_tokens = line[1:]
            # filter with BERT
            filtered_sentence_tokens = filter_with_bert_tokenizer(tokenizer, sentence_tokens)
            if len(sentence_tokens) != len(filtered_sentence_tokens):
                num_sentences_with_filtered_tokens += 1

            lines.append(f"{label} {' '.join(filtered_sentence_tokens)}\n")
            token_labels.append(f"{' '.join(['0' for _ in range(len(filtered_sentence_tokens))])}\n")

    print(f"Number of sentences with filtered tokens: {num_sentences_with_filtered_tokens} out of {len(lines)}")
    return lines, token_labels


def filter_with_bert_tokenizer(tokenizer, sentence_tokens):
    """
    Ignore tokens filtered by tokenizer
    :param tokenizer:
    :param sentence_tokens:
    :return: sentence_tokens with any original tokens filtered by tokenizer dropped
    """
    filtered_tokens = []

    for token in sentence_tokens:
        bert_tokens = tokenizer.tokenize(token)
        if len(bert_tokens) == 0:  # must be a special character filtered by BERT
            #pass
            print(f"Ignoring {token}")
        else:
            filtered_tokens.append(token)  # note that here we save the *original* token, not the WordPiece
    return filtered_tokens


def split_and_save(start_index, output_size, lines, token_labels, output_dir, output_filename_prefix, split_name):

    datasplit_lines = lines[start_index:start_index+output_size]
    datasplit_token_labels = token_labels[start_index:start_index+output_size]

    save_lines(path.join(output_dir, f"{output_filename_prefix}_{split_name}_size{output_size}.txt"), datasplit_lines)
    save_lines(path.join(output_dir, f"{output_filename_prefix}_{split_name}_labels_size{output_size}.txt"), datasplit_token_labels)

    return start_index+output_size


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_binaryevalformat_file', type=str, help="Input binaryevalformat file")
    parser.add_argument('--output_train_sizes', type=str, default="50000,200000",
                        help="Training set sizes; mutually exclusive remaining excluding dev and test is also saved;"
                             "Note that the smaller training sets are all subsets of the larger sets.")
    parser.add_argument('--output_dev_size', type=int, default=2000, help="output_dev_size")
    parser.add_argument('--output_test_size', type=int, default=2000, help="output_test_size")
    parser.add_argument('--seed_value', type=int, default=1776, help="seed_value")
    parser.add_argument('--output_dir', type=str, help="output_dir")
    parser.add_argument('--output_filename_prefix', type=str, help="filename followed by size")

    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)
    lines, token_labels = get_lines(args.input_binaryevalformat_file, tokenizer)

    np_random_state = np.random.RandomState(args.seed_value)
    # shuffle, and then split
    lines, token_labels = shuffle(lines, token_labels, random_state=np_random_state)

    # test:
    start_index = 0
    start_index = split_and_save(start_index, args.output_test_size, lines, token_labels, args.output_dir,
                                 args.output_filename_prefix, "test")

    # dev:
    start_index = split_and_save(start_index, args.output_dev_size, lines, token_labels, args.output_dir,
                                 args.output_filename_prefix, "dev")

    # training sizes
    output_train_sizes = [int(x) for x in args.output_train_sizes.split(',')]
    assert np.max(output_train_sizes) <= len(lines[start_index:])
    output_train_sizes.append(len(lines[start_index:]))
    training_start_index = start_index  # the smaller sets are subsets of the larger sets
    for output_train_size in output_train_sizes:
        _ = split_and_save(training_start_index, output_train_size, lines, token_labels, args.output_dir,
                                     args.output_filename_prefix, "train")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


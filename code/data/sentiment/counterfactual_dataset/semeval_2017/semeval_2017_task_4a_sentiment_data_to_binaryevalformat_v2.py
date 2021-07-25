# -*- coding: utf-8 -*-
"""
This scripts convert the SemEval-2017 Task 4A test data
(http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip) and converts it to the binary
prediction data format:

0 :: negative sentiment
1 :: positive sentiment

The input sentences are filtered with the BERT tokenizer to eliminate tokens that are elided by the tokenizer to avoid
downstream mis-matches with label alignments.

Also, note that to use the accuracy measure of past work, we balance the data set.

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

from pytorch_pretrained_bert.tokenization import BertTokenizer

from sklearn.utils import shuffle

random.seed(1776)

CLASS_0_LABEL="negative"
CLASS_1_LABEL="positive"



def get_lines(filepath_with_name, expected_number_of_fields, tokenizer):
    lines_neg = []
    labels_neg = []
    lines_pos = []
    labels_pos = []

    lines = []
    labels = []
    row_id = 0
    text_lens = []
    neg = 0
    pos = 0
    num_neutral = 0  # these are skipped for our binary setup
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            estimated_number_of_fields = row.count("\t") + 1  # tab separates the fields
            assert estimated_number_of_fields == expected_number_of_fields, \
                f"ERROR: The number of fields ({estimated_number_of_fields}) differs from the expected number ({expected_number_of_fields}) in row {row_id}."
            row = row.split()  # this will also get rid of internal newlines
            tweet_id = int(row[0])  # will throw an error if not parsable
            assert tweet_id > 0
            sentiment = row[1]
            assert sentiment in [CLASS_0_LABEL, CLASS_1_LABEL, "neutral"]
            tokens = row[2:]

            # there's no header line in this case
            if sentiment in [CLASS_0_LABEL, CLASS_1_LABEL]:
                if sentiment == CLASS_0_LABEL:
                    class_label = 0
                    neg += 1
                elif sentiment == CLASS_1_LABEL:
                    class_label = 1
                    pos += 1
                else:
                    assert False

                tokens, seq_labels = filter_with_bert_tokenizer(tokenizer, tokens, [f"{class_label}"] * len(tokens))
                text_lens.append(len(tokens))
                text = " ".join(tokens)
                lines.append(f"{class_label} {text}\n")
                labels.append(f"{' '.join(seq_labels)}\n")

                if sentiment == CLASS_0_LABEL:
                    lines_neg.append(f"{class_label} {text}\n")
                    labels_neg.append(f"{' '.join(seq_labels)}\n")
                elif sentiment == CLASS_1_LABEL:
                    lines_pos.append(f"{class_label} {text}\n")
                    labels_pos.append(f"{' '.join(seq_labels)}\n")
            else:
                num_neutral += 1
            row_id += 1
    print(f"Negative count: {neg}; Positive count: {pos}")
    print(f"Neutral count (ignored): {num_neutral}")
    print(f"Mean length: {np.mean(text_lens)}; std: {np.std(text_lens)}; min: {np.min(text_lens)}, max: {np.max(text_lens)}")
    full_set_mean = np.mean(text_lens)
    full_set_std = np.std(text_lens)
    assert len(lines_neg) == neg
    assert len(lines_pos) == pos
    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)
    balanced_lines = []
    balanced_labels = []
    if neg > pos:
        print(f"Reducing number of negative lines to {pos} in order to match the number of positive class instances.")
        lines_neg, labels_neg = shuffle(lines_neg, labels_neg, random_state=np_random_state)
        lines_neg = lines_neg[0:pos]
        labels_neg = labels_neg[0:pos]
    elif pos > neg:
        print(f"Reducing number of positive lines to {neg} in order to match the number of negative class instances.")
        lines_pos, labels_pos = shuffle(lines_pos, labels_pos, random_state=np_random_state)
        lines_pos = lines_pos[0:neg]
        labels_pos = labels_pos[0:neg]
    else:
        print(f"The number of positive and negative lines already match, so no re-balancing is necessary.")
    assert len(lines_neg) == len(lines_pos)
    assert len(labels_neg) == len(labels_pos)
    balanced_lines.extend(lines_neg)
    balanced_lines.extend(lines_pos)
    balanced_labels.extend(labels_neg)
    balanced_labels.extend(labels_pos)
    return lines, labels, balanced_lines, balanced_labels, full_set_mean, full_set_std




def filter_with_bert_tokenizer(tokenizer, sentence_tokens, token_labels):
    assert len(sentence_tokens) == len(token_labels)
    filtered_tokens = []
    filtered_token_labels = []

    for token, label in zip(sentence_tokens, token_labels):
        bert_tokens = tokenizer.tokenize(token)
        if len(bert_tokens) == 0:  # must be a special character filtered by BERT
            pass
            #print(f"Ignoring {token} with label {label}")
        else:
            filtered_tokens.append(token)
            filtered_token_labels.append(label)

    assert len(filtered_tokens) == len(filtered_token_labels)
    return filtered_tokens, filtered_token_labels


def print_x_sigma_length_lines(lines, full_set_mean, full_set_std, split_label):
    print(f"Checking {split_label}")
    for i, line in enumerate(lines):
        line = line.strip().split()
        #label = line[0]
        document = line[1:]
        if len(document) > full_set_mean + 2*full_set_std:
            print(f"{i} (len: {len(document)}): { ' '.join(document)}")

def check_lines(output_lines, output_labels_lines, output_balanced_lines, output_balanced_labels, full_set_mean, full_set_std):
    print_x_sigma_length_lines(output_lines, full_set_mean, full_set_std, "all lines")
    print_x_sigma_length_lines(output_balanced_lines, full_set_mean, full_set_std, "balanced lines")

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_tsv_file', type=str, help="input_tsv_file")
    parser.add_argument('--expected_number_of_fields', type=int, help="number of expected fields in the .tsv input")
    parser.add_argument('--output_binaryevalformat_file', type=str, help="output_binaryevalformat_file")
    parser.add_argument('--output_monolabels_file', type=str, help="placeholder labels (here, always match sentence labels)")

    parser.add_argument('--output_balanced_filtering_binaryevalformat_file', type=str, help="output_balanced_filtering_binaryevalformat_file")
    parser.add_argument('--output_balanced_filtering_monolabels_file', type=str, help="placeholder labels (here, always match sentence labels)")
    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    expected_number_of_fields = args.expected_number_of_fields
    assert args.input_tsv_file.strip() != args.output_binaryevalformat_file.strip()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)

    output_lines, output_labels_lines, output_balanced_lines, output_balanced_labels, full_set_mean, full_set_std = get_lines(args.input_tsv_file, expected_number_of_fields, tokenizer)
    # additional checks on data to check that there are no concatenated lines (due to errant newlines/etc. in the Tweets)
    check_lines(output_lines, output_labels_lines, output_balanced_lines, output_balanced_labels, full_set_mean, full_set_std)

    assert len(output_lines) == len(output_labels_lines)
    print(f"Length of output: {len(output_lines)}")
    save_lines(args.output_binaryevalformat_file, output_lines)
    save_lines(args.output_monolabels_file, output_labels_lines)

    print(f"Length of balanced output: {len(output_balanced_lines)}")
    save_lines(args.output_balanced_filtering_binaryevalformat_file, output_balanced_lines)
    save_lines(args.output_balanced_filtering_monolabels_file, output_balanced_labels)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


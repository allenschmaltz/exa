# -*- coding: utf-8 -*-
"""
This version is used with the Contrast Sets IMDb data.

Convert the Contrast Sets IMDb sentiment data (https://github.com/allenai/contrast-sets/tree/master/IMDb/data)
 to the binary prediction data format:

0 :: negative sentiment
1 :: positive sentiment

The input sentences are filtered with the BERT tokenizer to eliminate tokens that are elided by the tokenizer to avoid
downstream mis-matches with label alignments.

Here, we also check that the original test data matches the counterfactually-augmented test set formatting (e.g.,
quotation escapes should be handled the same). Note that the Contrast Sets data is double quote escaped. Also,
note that certain special characters are dropped in the Contrast Sets data, so 26 (out of 488) of the originals are
not verbatim matches with the corresponding originals in the counterfactually-augmented repo.

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

from pytorch_pretrained_bert.tokenization import BertTokenizer

random.seed(1776)

CLASS_0_LABEL= "Negative"
CLASS_1_LABEL= "Positive"

def get_original_lines(filepath_with_name):
    """

    """
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            lines.append(f"{line}\n")
    return lines

def get_lines(filepath_with_name, expected_number_of_fields, tokenizer, temp_file):
    # read in as tsv, since text field is quote escaped
    lines = []
    labels = []
    row_id = 0
    text_lens = []
    neg = 0
    pos = 0
    with open(filepath_with_name, newline='') as csvfile:
        csv_reader_obj = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in csv_reader_obj:
            assert len(row) == expected_number_of_fields, \
                f"ERROR: The number of fields ({len(row)}) differs from the expected number ({expected_number_of_fields})."
            sentiment = row[0]
            text = row[1]

            if row_id == 0:
                assert sentiment == "Sentiment" and text == "Text", f"{text}"
            else:
                # The Contrast Sets formatting includes an additional layer of quotes that need to be escaped.
                # For simplicity, we simply re-save (to a temp file) after removing the first layer of quotes and then re-open
                # and escape the top layer of quotes.
                with codecs.open(temp_file, "w", encoding="utf-8") as f:
                    f.write(f"{sentiment}\t{text}\n")
                with open(temp_file, newline='') as temp_csvfile:
                    second_reader = csv.reader(temp_csvfile, delimiter='\t', quotechar='"')
                    text_escaped_row = next(second_reader)

                    assert len(text_escaped_row) == expected_number_of_fields, f"ERROR: Check formatting: {text_escaped_row}"

                    _sentiment = text_escaped_row[0]
                    assert sentiment == _sentiment
                    text = text_escaped_row[1]

                    tokens = text.split()
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
            row_id += 1
    print(f"Negative count: {neg}; Positive count: {pos}")
    print(f"Mean length: {np.mean(text_lens)}; std: {np.std(text_lens)}; min: {np.min(text_lens)}, max: {np.max(text_lens)}")
    return lines, labels



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


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_tsv_file', type=str, help="input_tsv_file")
    parser.add_argument('--expected_number_of_fields', type=int, help="number of expected fields in the .tsv input")
    parser.add_argument('--input_counterfactual_original_binaryevalformat_file', type=str, help="If provided, the output is checked to match this file.")
    parser.add_argument('--output_binaryevalformat_file', type=str, help="output_binaryevalformat_file")
    parser.add_argument('--output_monolabels_file', type=str, help="placeholder labels (here, always match sentence labels)")

    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    input_tsv_file = args.input_tsv_file.strip()
    expected_number_of_fields = args.expected_number_of_fields
    output_binaryevalformat_file = args.output_binaryevalformat_file.strip()
    output_monolabels_file = args.output_monolabels_file.strip()

    assert input_tsv_file != output_binaryevalformat_file

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)

    # output_binaryevalformat_file is used here as a temp file; it is later over-written below
    output_lines, output_labels_lines = get_lines(input_tsv_file, expected_number_of_fields, tokenizer, output_binaryevalformat_file)
    assert len(output_lines) == len(output_labels_lines)
    print(f"Length of output: {len(output_lines)}")

    if args.input_counterfactual_original_binaryevalformat_file.strip() != "":
        print(f"Checking against the original counterfactually augmented file.")
        remaining_output_lines = []
        original_lines = get_original_lines(args.input_counterfactual_original_binaryevalformat_file)
        assert len(output_lines) == len(original_lines)
        for line in output_lines:
            if line in original_lines:
                original_line_index = original_lines.index(line)
                del original_lines[original_line_index]
            else:
                remaining_output_lines.append(line)
        print(f"Number of unmatched output lines: {len(remaining_output_lines)}")
        print(f"Number of remaining original lines: {len(original_lines)}")

        # print(f"unmatched output lines: {remaining_output_lines}")
        # print(f"remaining original lines: {original_lines}")

    save_lines(output_binaryevalformat_file, output_lines)
    save_lines(output_monolabels_file, output_labels_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


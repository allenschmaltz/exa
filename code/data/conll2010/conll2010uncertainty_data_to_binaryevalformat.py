# -*- coding: utf-8 -*-
"""
This version is used with the CONLL2010 uncertainty dataset. The original Shared Task data was not available at the
time of the experiments, so this is processed from the Szeged Uncertainty Corpus (https://rgai.sed.hu/node/160,
https://rgai.sed.hu/file/139), which is along the lines of that of the original Shared Task
(https://rgai.sed.hu/node/118#overlay-context=node/105). We have split the data by documents, to guard against
overlap across eval and train. We split off 10% of all documents for use as dev, and another 10% for the held-out test.

Here, we simply define the task as determining whether or not the sentence has at least 1 ccue tag. These tags
correspond to annotations of uncertain words, as defined further in the above links.:

0 :: certain sentence
1 :: uncertain sentence (i.e., has 1 or more ccue tags in the XML)

The input sentences are filtered with the BERT tokenizer to eliminate tokens that are elided by the tokenizer to
avoid downstream mis-matches with label alignments. (However, in this case, there should not be any special symbols
if using the data from the above link.)


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

from bs4 import BeautifulSoup
from sklearn.utils import shuffle

from pytorch_pretrained_bert.tokenization import BertTokenizer

random.seed(1776)


def get_lines(filepath_with_name, tokenizer):
    num_sentences = 0
    num_docs = 0
    # we group by documents, because we want to create train/dev/test splits from disjoint documents
    formatted_documents = []  # each document is a list of [formatted_line, formatted_label_line] pairs

    number_of_class0_tokens = 0
    number_of_class1_tokens = 0
    total_tokens = 0

    number_of_class0 = 0
    number_of_class1 = 0
    text_lens = []

    with open(filepath_with_name) as fp:
        soup = BeautifulSoup(fp, "xml")
        # split by document
        documents = soup.find_all("Document")
        for doc in documents:
            included_document = False
            document_parts = doc.find_all("DocumentPart")
            one_formatted_document = []
            for doc_part in document_parts:
                # we ignore the titles, headings, legend text, etc., except for factbank.xml where the distinction
                # is not made in the xml
                if filepath_with_name.endswith("factbank.xml"):
                    assert doc_part["type"] == "unknown"

                if (doc_part["type"] == "Text" or doc_part["type"] == "AbstractText") or \
                        (filepath_with_name.endswith("factbank.xml")):
                    sentences = doc_part.find_all("Sentence")
                    for sentence in sentences:
                        # the processed text should generally match the following (in some cases, there are minor
                        # whitespace or tokenization differences):
                        # full_text_of_sentence = sentence.get_text()
                        sent_tokens = []
                        sent_labels = []
                        uncertain = False
                        for sent_part in sentence:
                            if sent_part.name is None:
                                for token in sent_part.split():
                                    sent_tokens.append(token)
                                    sent_labels.append(0)
                                    number_of_class0_tokens += 1

                            elif sent_part.name == "ccue":
                                for token in sent_part.text.split():
                                    sent_tokens.append(token)
                                    sent_labels.append(1)
                                    uncertain = True
                                    number_of_class1_tokens += 1
                            else:
                                assert False, f"ERROR: Not expecting further nesting within the sentence"
                        if len(sent_tokens) != 0:
                            included_document = True
                            num_sentences += 1

                            class_label = int(uncertain)
                            if class_label == 0:
                                number_of_class0 += 1
                            else:
                                number_of_class1 += 1

                            tokens, seq_labels = filter_with_bert_tokenizer(tokenizer, sent_tokens, sent_labels)
                            if len(tokens) != len(sent_tokens):
                                print(f"WARNING: An unexpected character was seen by the BERT tokenizer, so dropping:"
                                      f"\n\tORIGINAL: {' '.join(sent_tokens)}"
                                      f"\n\tCLEANED: {' '.join(tokens)}"
                                      f"\n\tCLEANED: {' '.join([str(x) for x in seq_labels])}")

                            total_tokens += len(tokens)
                            text_lens.append(len(tokens))
                            formatted_line = f"{class_label} {' '.join(tokens)}\n"
                            formatted_label_line = f"{' '.join([str(x) for x in seq_labels])}\n"

                            one_formatted_document.append([formatted_line, formatted_label_line])

            if included_document:
                assert len(one_formatted_document) != 0
                formatted_documents.append(one_formatted_document)
                num_docs += 1
    print(f"Total number of sentences: {num_sentences}")
    print(f"Total number of documents: {num_docs}")

    print(f"Class 0 count: {number_of_class0}; Class 1 count: {number_of_class1}")
    print(f"Mean length: {np.mean(text_lens)}; std: {np.std(text_lens)};"
          f" min: {np.min(text_lens)}, max: {np.max(text_lens)}")
    print(f"Number of class 0 tokens: {number_of_class0_tokens}")
    print(f"Number of class 1 tokens: {number_of_class1_tokens}")

    print(f"Total tokens: {total_tokens}")
    return formatted_documents


def filter_with_bert_tokenizer(tokenizer, sentence_tokens, token_labels):
    assert len(sentence_tokens) == len(token_labels)
    filtered_tokens = []
    filtered_token_labels = []

    for token, label in zip(sentence_tokens, token_labels):
        bert_tokens = tokenizer.tokenize(token)
        if len(bert_tokens) == 0:  # must be a special character filtered by BERT
            print(f"Ignoring {token} with label {label}")
            assert False, f"ERROR: No special symbols are expected in this dataset."
        else:
            filtered_tokens.append(token)
            filtered_token_labels.append(label)

    assert len(filtered_tokens) == len(filtered_token_labels)
    return filtered_tokens, filtered_token_labels


def flatten_documents(docs, split_label):
    # take lists ('docs') of pairs of [formatted_lines, formatted_label_lines] and return two lists:
    # [formatted_lines, ...]
    # [formatted_label_lines, ...]
    formatted_lines = []
    formatted_label_lines = []
    for doc in docs:
        for sentence_labels_pair in doc:
            sentence, sentence_labels = sentence_labels_pair
            formatted_lines.append(sentence)
            formatted_label_lines.append(sentence_labels)
    assert len(formatted_lines) == len(formatted_label_lines)
    print(f"Total sentences in {split_label}: {len(formatted_lines)}")
    return formatted_lines, formatted_label_lines


def create_splits(all_docs, np_random_state, eval_split_size_proportion):
    shuffled_all_docs = shuffle(all_docs, random_state=np_random_state)
    assert shuffled_all_docs != all_docs
    # split by documents:
    eval_size = int(eval_split_size_proportion*len(shuffled_all_docs))
    test_docs = shuffled_all_docs[0:eval_size]
    dev_docs = shuffled_all_docs[eval_size:eval_size*2]
    train_docs = shuffled_all_docs[eval_size*2:]

    formatted_lines_test, formatted_label_lines_test = flatten_documents(test_docs, "test")
    formatted_lines_dev, formatted_label_lines_dev = flatten_documents(dev_docs, "dev")
    formatted_lines_train, formatted_label_lines_train = flatten_documents(train_docs, "train")

    return formatted_lines_train, formatted_label_lines_train, \
           formatted_lines_dev, formatted_label_lines_dev, \
           formatted_lines_test, formatted_label_lines_test


def remove_test_overlap(output_lines_train, output_labels_lines_train,
                      output_lines_dev, output_labels_lines_dev,
                      output_lines_test, output_labels_lines_test):
    # check that the test sentences do not appear verbatim in train or dev, and if they do, remove them from test
    filtered_output_lines_test = []
    filtered_output_labels_lines_test = []
    for test_sentence, test_sentence_labels in zip(output_lines_test, output_labels_lines_test):
        found = False
        if test_sentence in output_lines_dev:
            print(f"The following test sentence also appears in the dev split, so it is removed from test: "
                  f"\n{test_sentence}")
            found = True
        if test_sentence in output_lines_train:
            print(f"The following test sentence also appears in the train split, so it is removed from test: "
                  f"\n{test_sentence}")
            found = True
        if not found:
            filtered_output_lines_test.append(test_sentence)
            filtered_output_labels_lines_test.append(test_sentence_labels)

    return filtered_output_lines_test, filtered_output_labels_lines_test


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_files', type=str, help="input_files")
    parser.add_argument('--seed_value', default=1776, type=int, help="seed_value")
    parser.add_argument('--eval_split_size_proportion', default=0.1, type=float,
                        help="Proportion of all *documents* to use for each of dev and test. The remaining documents "
                             "become the training set.")
    parser.add_argument('--output_binaryevalformat_train_file', type=str,
                        help="output_binaryevalformat_train_file")
    parser.add_argument('--output_binaryevalformat_train_labels_file', type=str,
                        help="output_binaryevalformat_train_labels_file")
    parser.add_argument('--output_binaryevalformat_dev_file', type=str,
                        help="output_binaryevalformat_dev_file")
    parser.add_argument('--output_binaryevalformat_dev_labels_file', type=str,
                        help="output_binaryevalformat_dev_labels_file")
    parser.add_argument('--output_binaryevalformat_test_file', type=str,
                        help="output_binaryevalformat_test_file")
    parser.add_argument('--output_binaryevalformat_test_labels_file', type=str,
                        help="output_binaryevalformat_test_labels_file")

    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)

    np_random_state = np.random.RandomState(args.seed_value)
    input_files = args.input_files.split(",")
    all_docs = []
    for input_file in input_files:
        docs = get_lines(input_file, tokenizer)
        all_docs.extend(docs)
    print(f"------------------------")
    print(f"Total number of documents: {len(all_docs)}")

    output_lines_train, output_labels_lines_train, \
        output_lines_dev, output_labels_lines_dev, \
        output_lines_test, output_labels_lines_test = \
        create_splits(all_docs, np_random_state, args.eval_split_size_proportion)

    output_lines_test, output_labels_lines_test = \
        remove_test_overlap(output_lines_train, output_labels_lines_train,
                            output_lines_dev, output_labels_lines_dev,
                            output_lines_test, output_labels_lines_test)
    print(f"Total filtered sentences in test: {len(output_lines_test)}")

    save_lines(args.output_binaryevalformat_train_file, output_lines_train)
    save_lines(args.output_binaryevalformat_train_labels_file, output_labels_lines_train)

    save_lines(args.output_binaryevalformat_dev_file, output_lines_dev)
    save_lines(args.output_binaryevalformat_dev_labels_file, output_labels_lines_dev)

    save_lines(args.output_binaryevalformat_test_file, output_lines_test)
    save_lines(args.output_binaryevalformat_test_labels_file, output_labels_lines_test)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


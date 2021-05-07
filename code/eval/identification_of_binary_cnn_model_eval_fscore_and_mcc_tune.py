# -*- coding: utf-8 -*-
"""
Tune (for identification) the output from the binary cnn models. F1, F0.5, MCC

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
import torch

random.seed(1776)


ID_CORRECT = 0 # "negative" class
ID_WRONG = 1  # "positive class"

def get_labels_and_scores(filepath_with_name):
    labels = []
    total_errors = 0
    probs = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            label = int(line[0])
            labels.append(label)
            logit = torch.tensor([float(line[1]), float(line[2])])
            prob = torch.nn.functional.softmax(logit, dim=0)
            probs.append(prob)
            if label == 1:
                total_errors += 1

    print("Percent of gold sentences with errors: {} ({} out of {})".format(total_errors/float(len(labels)),
                                                                       total_errors, len(labels)))
    return labels, probs


def calculate_metrics(gold_labels, predicted_labels):
    assert len(predicted_labels) == len(gold_labels)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # ID_CORRECT = 0 # "negative" class
    # ID_WRONG = 1  # "positive class"

    for pred, gold in zip(predicted_labels, gold_labels):
        if gold == ID_WRONG:
            if pred == ID_WRONG:
                tp += 1
            elif pred == ID_CORRECT:
                fn += 1
        elif gold == ID_CORRECT:
            if pred == ID_WRONG:
                fp += 1
            elif pred == ID_CORRECT:
                tn += 1

    precision = tp / float(tp + fp) if tp + fp > 0 else 0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0

    def fscore(beta, precision, recall):
        return (1 + beta**2) * (precision*recall) / float(beta**2*precision + recall) if float(beta**2*precision + recall) > 0 else 0

    print("\tPrecision: {}".format(precision))
    print("\tRecall: {}".format(recall))
    print("\tF1: {}".format(fscore(1.0, precision, recall)))
    print("\tF0.5: {}".format(fscore(0.5, precision, recall)))


    # Calculate MCC:

    denominator = np.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )

    if denominator == 0.0:
        print("\tWarning: denominator in mcc calculation is 0; setting denominator to 1.0")
        denominator = 1.0

    mcc = (tp*tn - fp*fn) / float(denominator)

    print("\tMCC: {}".format(mcc))

def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_score_vals_file', help="Output from the cnn model (include gold labels)")
    parser.add_argument('--start_offset', type=float, default=0.0, help="start_offset")
    parser.add_argument('--end_offset', type=float, default=1.0, help="end_offset")
    parser.add_argument('--number_of_offsets', type=int, default=50, help="number_of_offsets")


    args = parser.parse_args(arguments)

    input_score_vals_file = args.input_score_vals_file
    start_offset = args.start_offset
    end_offset = args.end_offset
    number_of_offsets = args.number_of_offsets

    gold_labels, probs = get_labels_and_scores(input_score_vals_file)

    ## random assignment:
    print("--" * 30)
    print("RANDOM")
    predicted_labels = []
    pred_error_count = 0
    for _ in probs:
        i = [0, 1]
        random.shuffle(i)
        pred = i[0]
        predicted_labels.append(pred)
        if pred == 1:
            pred_error_count += 1

    print(f"\tPredicted error count: {pred_error_count}")

    calculate_metrics(gold_labels, predicted_labels)

    ## most likely assignment:
    print("--" * 30)
    print("MAJORITY CLASS")
    predicted_labels = []
    pred_error_count = 0
    for _ in probs:
        pred = 1
        predicted_labels.append(pred)
        if pred == 1:
            pred_error_count += 1

    print(f"\tPredicted error count: {pred_error_count}")

    calculate_metrics(gold_labels, predicted_labels)

    for offset in np.linspace(start_offset, end_offset, number_of_offsets): #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print("--" * 30)
        print(f"Offset: {offset}")
        predicted_labels = []
        pred_error_count = 0
        for prob_0_1 in probs:
            prob = torch.tensor(prob_0_1)
            prob[1] += offset
            pred = prob.argmax().item()

            predicted_labels.append(pred)
            if pred == 1:
                pred_error_count += 1

        print(f"\tPredicted error count: {pred_error_count}")

        calculate_metrics(gold_labels, predicted_labels)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


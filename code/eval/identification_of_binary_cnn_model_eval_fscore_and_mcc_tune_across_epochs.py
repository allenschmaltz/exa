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

def get_labels_and_scores(filepath_with_name, print_output=False):
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
    if print_output:
        print("Percent of gold sentences with errors: {} ({} out of {})".format(total_errors/float(len(labels)),
                                                                       total_errors, len(labels)))
    return labels, probs


def fscore(beta, precision, recall):
    return (1 + beta**2) * (precision*recall) / float(beta**2*precision + recall)

def calculate_metrics(gold_labels, predicted_labels, print_output=False):
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

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    f1 = fscore(1.0, precision, recall)
    f0_5 = fscore(0.5, precision, recall)

    if print_output:
        print("\tPrecision: {}".format(precision))
        print("\tRecall: {}".format(recall))
        print("\tF1: {}".format(f1))
        print("\tF0.5: {}".format(f0_5))


    # Calculate MCC:

    denominator = np.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )

    if denominator == 0.0:
        if print_output:
            print("\tWarning: denominator in mcc calculation is 0; setting denominator to 1.0")
        denominator = 1.0

    mcc = (tp*tn - fp*fn) / float(denominator)

    if print_output:
        print("\tMCC: {}".format(mcc))

    return f1, f0_5, mcc

def update_stats(metric, max_metric, max_metric_offset, offset, epoch, max_epoch):
    if metric > max_metric:
        max_metric = metric
        max_metric_offset = offset
        max_epoch = epoch
    return max_metric, max_metric_offset, max_epoch

def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_score_vals_file_prefix', help="Prefix (excluding epochs) of scores file")
    parser.add_argument('--start_epoch', type=int, help="start_epoch")
    parser.add_argument('--end_epoch', type=int, help="end_epoch")

    args = parser.parse_args(arguments)

    input_score_vals_file_prefix = args.input_score_vals_file_prefix
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    # get gold labels from the first epoch
    input_score_vals_file = f"{input_score_vals_file_prefix}.epoch{start_epoch}.txt"
    gold_labels, probs = get_labels_and_scores(input_score_vals_file, True)

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

    calculate_metrics(gold_labels, predicted_labels, True)

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

    calculate_metrics(gold_labels, predicted_labels, True)

    max_f1_epoch = -1
    max_f1_offset = 0
    max_f1 = 0
    max_0_5_epoch = -1
    max_f0_5_offset = 0
    max_f0_5 = 0
    max_mcc_epoch = -1
    max_mcc_offset = 0
    max_mcc = 0

    # loop through epochs

    for epoch in range(start_epoch, end_epoch+1):
        print(f"Epoch {epoch}")
        input_score_vals_file = f"{input_score_vals_file_prefix}.epoch{epoch}.txt"
        gold_labels, probs = get_labels_and_scores(input_score_vals_file)

        for offset in np.linspace(0, 1.0, 50):
            #print("--" * 30)
            #print(f"Offset: {offset}")
            predicted_labels = []
            pred_error_count = 0
            for prob_0_1 in probs:
                prob = torch.tensor(prob_0_1)
                prob[1] += offset
                pred = prob.argmax().item()

                predicted_labels.append(pred)
                if pred == 1:
                    pred_error_count += 1

            #print(f"\tPredicted error count: {pred_error_count}")

            f1, f0_5, mcc = calculate_metrics(gold_labels, predicted_labels)

            max_f1, max_f1_offset, max_f1_epoch = update_stats(f1, max_f1, max_f1_offset, offset, epoch, max_f1_epoch)
            max_f0_5, max_f0_5_offset, max_0_5_epoch = update_stats(f0_5, max_f0_5, max_f0_5_offset, offset, epoch, max_0_5_epoch)
            max_mcc, max_mcc_offset, max_mcc_epoch = update_stats(mcc, max_mcc, max_mcc_offset, offset, epoch, max_mcc_epoch)

    print(f"Max F1: {max_f1}; epoch {max_f1_epoch} offset {max_f1_offset}")
    print(f"Max F0.5: {max_f0_5}; epoch {max_0_5_epoch} offset {max_f0_5_offset}")
    print(f"Max MCC: {max_mcc}; epoch {max_mcc_epoch} offset {max_mcc_offset}")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


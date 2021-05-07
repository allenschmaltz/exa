import numpy as np

import utils
import constants

# def calculate_fscore_from_stats(tp, fp, fn, beta):
#     precision = tp / float(tp + fp) if tp + fp > 0 else 0
#     recall = tp / float(tp + fn) if tp + fn > 0 else 0
#
#
#     def fscore(beta, precision, recall):
#         return (1 + beta ** 2) * (precision * recall) / float(beta ** 2 * precision + recall) if float(
#             beta ** 2 * precision + recall) > 0 else 0
#
#     return precision, recall, fscore(beta, precision, recall)

def calculate_metrics(gold_labels, predicted_labels, numerically_stable, print_results=True):
    assert len(predicted_labels) == len(gold_labels)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # ID_CORRECT = 0 # "negative" class
    # ID_WRONG = 1  # "positive class"

    for pred, gold in zip(predicted_labels, gold_labels):
        if gold == constants.ID_WRONG:
            if pred == constants.ID_WRONG:
                tp += 1
            elif pred == constants.ID_CORRECT:
                fn += 1
        elif gold == constants.ID_CORRECT:
            if pred == constants.ID_WRONG:
                fp += 1
            elif pred == constants.ID_CORRECT:
                tn += 1

    precision = tp / float(tp + fp) if tp + fp > 0 else 0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0

    def fscore(beta, precision, recall):
        return (1 + beta**2) * (precision*recall) / float(beta**2*precision + recall) if float(beta**2*precision + recall) > 0 else 0

    fscore_1 = fscore(1.0, precision, recall)
    fscore_0_5 = fscore(0.5, precision, recall)

    if print_results:
        print("\tPrecision: {}".format(precision))
        print("\tRecall: {}".format(recall))
        print("\tF1: {}".format(fscore_1))
        print("\tF0.5: {}".format(fscore_0_5))

    # Calculate MCC:
    if numerically_stable:
        denominator = np.exp(0.5 * (np.log(tp+fp) + np.log(tp+fn) + np.log(tn+fp) + np.log(tn+fn)))
    else:
        denominator = np.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )
    if denominator == 0.0:
        print("\tWarning: denominator in mcc calculation is 0; setting denominator to 1.0")
        denominator = 1.0
    mcc = (tp*tn - fp*fn) / float(denominator)
    if print_results:
        print("\tMCC: {}".format(mcc))

    return {"precision": precision, "recall": recall, "fscore_1": fscore_1, "fscore_0_5": fscore_0_5, "mcc": mcc}


def calculate_metrics_window(gold_labels_per_sentence, predicted_labels_per_sentence, window_width, numerically_stable):
    assert len(predicted_labels_per_sentence) == len(gold_labels_per_sentence)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # ID_CORRECT = 0 # "negative" class
    # ID_WRONG = 1  # "positive class"

    for predicted_labels, gold_labels in zip(predicted_labels_per_sentence, gold_labels_per_sentence):
        assert len(predicted_labels) == len(gold_labels)
        for i, (_pred, _gold) in enumerate(zip(predicted_labels, gold_labels)):
            pred_window = predicted_labels[max(0, i-window_width):min(i+window_width+1, len(predicted_labels))]
            gold_window = gold_labels[max(0, i-window_width):min(i+window_width+1, len(gold_labels))]

            if constants.ID_WRONG in pred_window:
                pred = constants.ID_WRONG
            else:
                pred = constants.ID_CORRECT
            if ID_WRONG in gold_window:
                gold = constants.ID_WRONG
            else:
                gold = constants.ID_CORRECT

            if gold == constants.ID_WRONG:
                if pred == constants.ID_WRONG:
                    tp += 1
                elif pred == constants.ID_CORRECT:
                    fn += 1
            elif gold == constants.ID_CORRECT:
                if pred == constants.ID_WRONG:
                    fp += 1
                elif pred == constants.ID_CORRECT:
                    tn += 1

            # if gold == ID_WRONG:
            #     if ID_WRONG in pred_window:
            #     #if pred == ID_WRONG:
            #         tp += 1
            #     #elif pred == ID_CORRECT:
            #     else:
            #         fn += 1
            # elif gold == ID_CORRECT:
            #     if ID_WRONG in pred_window:
            #     #if pred == ID_WRONG:
            #         fp += 1
            #     #elif pred == ID_CORRECT:
            #     else:
            #         tn += 1

    precision = tp / float(tp + fp) if tp + fp > 0 else 0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0

    def fscore(beta, precision, recall):
        return (1 + beta**2) * (precision*recall) / float(beta**2*precision + recall) if float(beta**2*precision + recall) > 0 else 0

    print("\tPrecision: {}".format(precision))
    print("\tRecall: {}".format(recall))
    print("\tF1: {}".format(fscore(1.0, precision, recall)))
    print("\tF0.5: {}".format(fscore(0.5, precision, recall)))

    # Calculate MCC:
    if numerically_stable:
        denominator = np.exp(0.5 * (np.log(tp+fp) + np.log(tp+fn) + np.log(tn+fp) + np.log(tn+fn)))
    else:
        denominator = np.sqrt( (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) )
    if denominator == 0.0:
        print("\tWarning: denominator in mcc calculation is 0; setting denominator to 1.0")
        denominator = 1.0
    mcc = (tp*tn - fp*fn) / float(denominator)
    print("\tMCC: {}".format(mcc))


def calculate_seq_metrics(gold_lines, contribution_tuples_per_sentence, sentence_probs, tune_offset=True,
                          print_baselines=True, output_generated_detection_file="", numerically_stable=True,
                          fce_eval=True):
    if tune_offset:
        print(f"NOTE: The current version does not tune offsets for contributions that exactly equal 0.0, "
              f"since these always correspond to padding symbols in the convention/datasets used for the original "
              f"paper.")

    if tune_offset:
        #offset_list = list(np.linspace(-10, 10, 201)) + [0.0]
        offset_list = list(np.linspace(-2.5, 2.5, 51)) + [0.0]  # true 0.0 is always calculated before the return
    else:
        offset_list = [0.0]
    for detection_offset in offset_list:  # [-1, -2, 0, 1, 2]:
        # print("-------------------------------------------------------")
        # print(f"DETECTION OFFSET: {detection_offset}")
        all_gold_labels = []
        all_generated_labels = []

        gold_labels_per_sentence = []
        generated_labels_per_sentence = []

        output_generated_detection = []
        # for gold, generated in zip(gold_lines, generated_idx_per_sentence):
        for gold, generated, sentence_prob in zip(gold_lines, contribution_tuples_per_sentence, sentence_probs):
            neg_sentence_prob = sentence_prob[0]
            pos_sentence_prob = sentence_prob[1]

            if not fce_eval:
                gold_labels = convert_diffs_to_detection_labels(gold)
            else:
                gold_labels = [int(x) for x in gold]  # in this case, no final holder sym
            generated_labels = []
            for gradient_idx in generated:
                neg_val, pos_val, neg_logit_bias, pos_logit_bias = gradient_idx
                output_generated_detection.append(
                    f"{neg_val + neg_logit_bias}\t{pos_val + pos_logit_bias}\t{neg_sentence_prob}\t{pos_sentence_prob}\n")
                if neg_val == 0.0 and pos_val == 0.0:
                    # Note that true 0.0 is never tuned, because this only occurs with padding symbols in this version.
                    generated_labels.append(constants.ID_CORRECT)
                else:
                    if pos_val + pos_logit_bias > neg_val + neg_logit_bias + detection_offset:

                        generated_labels.append(constants.ID_WRONG)
                    else:
                        generated_labels.append(constants.ID_CORRECT)
            output_generated_detection.append(f"\n")

            assert len(gold_labels) == len(
                generated_labels), f"{len(gold_labels)} {len(generated_labels)}"  # f"{' '.join([str(x) for x in gold_labels])}; {' '.join([str(x) for x in generated_labels])}"

            all_gold_labels.extend(gold_labels)
            all_generated_labels.extend(generated_labels)

            gold_labels_per_sentence.append(gold_labels)
            generated_labels_per_sentence.append(generated_labels)

        if output_generated_detection_file != "":
            utils.save_lines(output_generated_detection_file, output_generated_detection)
            output_generated_detection_file = ""

        if print_baselines:
            print("RANDOM CLASS")
            calculate_metrics(all_gold_labels, np.random.choice(2, len(all_gold_labels)), numerically_stable)
            print("")
            print("MAJORITY CLASS")
            calculate_metrics(all_gold_labels, np.ones(len(all_gold_labels)), numerically_stable)
            print("")
            print_baselines = False

        if tune_offset:
            print("-------------------------------------------------------")
            print(f"DETECTION OFFSET: {detection_offset}")
            print("GENERATED:")
            calculate_metrics(all_gold_labels, all_generated_labels, numerically_stable)
    # return the null offset results; this assumes [0.0] is at end of list
    assert detection_offset == 0.0
    print("-------------------------------------------------------")
    print(f"DETECTION OFFSET: {detection_offset}")
    print("GENERATED:")
    return calculate_metrics(all_gold_labels, all_generated_labels, numerically_stable)
        # """
        # for window_wing_size in range(1, 4):
        #     print("")
        #     print(f"GENERATED WINDOW, SIZE: {window_wing_size*2+1}:")
        #     calculate_metrics_window(gold_labels_per_sentence, generated_labels_per_sentence, window_wing_size, numerically_stable)
        # """
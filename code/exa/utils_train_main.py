# These are the basic methods for training at the sentence-level, and for fine-tuning the CNN/memory layer with a
# min-max sparsity loss and for fine-tuning with fully-supervised labels. Note that here we always keep the BERT
# model frozen. To fine-tune the BERT model, iteratively freezing/unfreezing the BERT model and the memory layer
# (i.e., the uniCNN) is likely productive (see the memory matching repo and paper for an example of such training).

from model import CNN
import utils
import constants
import utils_transformer
import utils_eval
import utils_viz
import utils_exemplar
import utils_sequence_labeling
import utils_classification

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy

import math

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from collections import defaultdict

import torch.nn.functional as F

import codecs


def train(data, params, np_random_state, bert_model, bert_device, only_save_best_models=False):
    # This is for standard sentence-level training.
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format(params['word_embeddings_file'],
                                                         binary=not params["word_embeddings_file_in_plaintext"])
        wv_matrix = []
        # one for zero padding and one for unk
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        for i in range(params["vocab_size"]-2):
            idx = i+2
            word = data["idx_to_word"][idx]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    print("Initializing model")
    if params["GPU"] != -1:
        model = CNN(**params).cuda(params["GPU"])
    else:
        model = CNN(**params)

    print("Starting training")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_dev_acc_epoch = -1

    num_batch_instances = math.ceil( (len(data["idx_train_x"]) / params["BATCH_SIZE"]) )
    all_epoch_cumulative_losses = []
    for e in range(params["EPOCH"]):
        if bert_model is not None:
            data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"], data["train_y"] = \
                shuffle(data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"],
                        data["train_y"], random_state=np_random_state)
        else:
            data["idx_train_x"], data["train_y"] = shuffle(data["idx_train_x"], data["train_y"],
                                                           random_state=np_random_state)
        batch_num = 0
        cumulative_losses = []
        for i in range(0, len(data["idx_train_x"]), params["BATCH_SIZE"]):
            if batch_num % int(num_batch_instances * 0.25) == 0:
                print(f"Epoch {e+1}, {batch_num/num_batch_instances}")
                if len(cumulative_losses) > 0:
                    print(f"\tCurrent epoch average loss: {np.mean(cumulative_losses)}")
            batch_num += 1
            batch_range = min(params["BATCH_SIZE"], len(data["idx_train_x"]) - i)

            batch_x = data["idx_train_x"][i:i + batch_range]

            if bert_model is not None:
                # get BERT representations
                bert_output = \
                    utils_transformer.get_bert_representations(data["train_bert_idx_sentences"][i:i + batch_range],
                                                               data["train_bert_input_masks"][i:i + batch_range],
                                                               bert_model, bert_device, params["bert_layers"],
                                                               len(batch_x[0]))
            else:
                bert_output = None
            batch_y = data["train_y"][i:i + batch_range]

            if params["GPU"] != -1:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
                batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output))
                batch_y = Variable(torch.LongTensor(batch_y))

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x, bert_output, forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION)
            loss = criterion(pred, batch_y)
            cumulative_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        print(f"Epoch average loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")
        dev_acc, score_vals = utils_classification.test(data, model, params, bert_model, bert_device, mode="dev")

        print("epoch:", e + 1, "/ dev_acc:", dev_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc >= max_dev_acc:
            max_dev_acc = dev_acc
            max_dev_acc_epoch = e + 1
            if only_save_best_models:
                print(
                    f"Saving epoch {e + 1} as new best max_dev_acc_epoch model")
                utils.save_model_torch(model, params, f"max_dev_acc_epoch_epoch")

                print("Saving scores file")
                utils.save_lines(params["score_vals_file"] + f".epoch{e+1}.txt", score_vals)
                print(f"Saved scores file: {params['score_vals_file']}.epoch{e+1}.txt")

        print(f"\tCurrent max dev accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")

        # save after *every* epoch, unless only_save_best_models
        if e + 1 > 0:
            if params["SAVE_MODEL"] and not only_save_best_models:
                print(f"Saving epoch {e+1}")
                utils.save_model_torch(model, params, e + 1)
                print("Saving scores file")
                utils.save_lines(params["score_vals_file"] + f".epoch{e+1}.txt", score_vals)
                print(f"Saved scores file: {params['score_vals_file']}.epoch{e+1}.txt")

    print(f"Max dev accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")


def seq_labeling_fine_tune(options, model, data, params, np_random_state, bert_model, bert_device,
                           only_save_best_models=False):
    # This is intended for supervised token-level (a.k.a., sequence-level) training. In the current version,
    # this is intended to be used with a model that has already been initially trained at the sentence-level,
    # using, for example, train(). If --only_save_best_models, there's an option to either save by the
    # sequence-level metrics, or by the sentence-level accuracy, where a positive prediction occurs if at least
    # one sequence label is positive. Note that this latter option is primarily for debugging/reference; typically
    # if it's desired to make a sentence-level prediction from the local predictions, a global constraint should
    # be added. The multi-label extension to this binary setting explains how to do that. See that paper for further
    # details.
    print("Starting fine-tuning")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])

    seq_labeling_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    pre_dev_acc = 0
    max_dev_acc = 0
    max_dev_acc_epoch = -1

    num_batch_instances = math.ceil( (len(data["idx_train_x"]) / params["BATCH_SIZE"]) )
    all_epoch_cumulative_losses = []

    if options.eval_fine_tuning_epoch_based_on_local_to_global_accuracy:
        metric_label = "local_to_global_acc"
    else:
        metric_label = options.max_metric_label

    for e in range(params["EPOCH"]):
        if bert_model is not None:
            data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"], data["train_y"], \
            data["train_padded_seq_y"], data["train_padded_seq_y_mask"] = \
                shuffle(data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"],
                        data["train_y"], data["train_padded_seq_y"], data["train_padded_seq_y_mask"],
                        random_state=np_random_state)
        else:
            data["idx_train_x"], data["train_y"], data["train_padded_seq_y"], data["train_padded_seq_y_mask"] = \
                shuffle(data["idx_train_x"], data["train_y"], data["train_padded_seq_y"],
                        data["train_padded_seq_y_mask"], random_state=np_random_state)
        batch_num = 0
        cumulative_losses = []
        for i in range(0, len(data["idx_train_x"]), params["BATCH_SIZE"]):
            if batch_num % int(num_batch_instances * 0.25) == 0:
                print(f"Epoch {e+1}, {batch_num/num_batch_instances}")
                if len(cumulative_losses) > 0:
                    print(f"\tCurrent epoch average loss: {np.mean(cumulative_losses)}")
            batch_num += 1
            batch_range = min(params["BATCH_SIZE"], len(data["idx_train_x"]) - i)

            batch_x = data["idx_train_x"][i:i + batch_range]

            batch_train_padded_seq_y = \
                torch.FloatTensor(data["train_padded_seq_y"][i:i + batch_range]).to(params["main_device"])
            batch_train_padded_seq_y_mask = \
                torch.FloatTensor(data["train_padded_seq_y_mask"][i:i + batch_range]).to(params["main_device"])

            if bert_model is not None:
                # get BERT representations
                bert_output = \
                    utils_transformer.get_bert_representations(data["train_bert_idx_sentences"][i:i + batch_range],
                                                               data["train_bert_input_masks"][i:i + batch_range],
                                                               bert_model, bert_device, params["bert_layers"],
                                                               len(batch_x[0]))
            else:
                bert_output = None

            if params["GPU"] != -1:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output))

            optimizer.zero_grad()
            model.train()

            pred_seq_labels = model(batch_x, bert_output, forward_type=constants.FORWARD_TYPE_SEQUENCE_LABELING,
                                    main_device=params["main_device"])

            loss = seq_labeling_criterion(pred_seq_labels, batch_train_padded_seq_y)
            loss = batch_train_padded_seq_y_mask * loss
            loss = loss.sum() / batch_train_padded_seq_y_mask.sum()

            cumulative_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        print(f"Epoch average loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")

        global_acc_from_local, score_vals = utils_classification.test_based_on_contributions(data, model, params,
                                                                                             bert_model,
                                                                                             bert_device, mode="dev")

        print(f"Calculating sequence label metrics:")
        contribution_tuples_per_sentence, sentence_probs = utils_sequence_labeling.test_seq_labels(data, model,
                                                                                                   params, bert_model,
                                                                                                   bert_device,
                                                                                                   mode="dev",
                                                                                                   fce_eval=True)

        eval_stats = utils_eval.calculate_seq_metrics(data["dev_seq_y"], contribution_tuples_per_sentence,
                                                      sentence_probs, tune_offset=False,
                                                      print_baselines=False,
                                                      output_generated_detection_file="",
                                                      numerically_stable=True, fce_eval=True)
        if options.eval_fine_tuning_epoch_based_on_local_to_global_accuracy:
            print(f"Reference: The token-level {options.max_metric_label}: {eval_stats[options.max_metric_label]}")
            dev_acc = global_acc_from_local
        else:
            print(f"Reference: The sentence-level accuracy derived from the token-level logits: "
                  f"{global_acc_from_local}")
            dev_acc = eval_stats[options.max_metric_label]

        print("epoch:", e + 1, f"/ dev_{metric_label}:", dev_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc >= max_dev_acc:
            max_dev_acc = dev_acc
            max_dev_acc_epoch = e + 1
            if only_save_best_models:
                print(
                    f"Saving epoch {e + 1} as new best dev {metric_label} model, saved as "
                    f"max_dev_{metric_label}_epoch")
                # utils.save_model_torch(model, params, f"max_dev_acc_epoch_epoch")
                utils.save_model_torch(model, params, f"max_dev_{metric_label}_epoch")

                print("Saving local to global scores file (note this is primarily for debugging)")
                utils.save_lines(params["score_vals_file"] + f".epoch{e + 1}.txt", score_vals)
                print(f"Saved local to global scores file: {params['score_vals_file']}.epoch{e + 1}.txt")

        print(f"\tCurrent max dev {metric_label}: {max_dev_acc} at epoch {max_dev_acc_epoch}")

        # save after *every* epoch, unless only_save_best_models
        if e + 1 > 0:
            if params["SAVE_MODEL"] and not only_save_best_models:
                print(f"Saving epoch {e + 1}")
                utils.save_model_torch(model, params, e + 1)
                print("Saving local to global scores file (note this is primarily for debugging)")
                utils.save_lines(params["score_vals_file"] + f".epoch{e + 1}.txt", score_vals)
                print(f"Saved local to global scores file: {params['score_vals_file']}.epoch{e + 1}.txt")

    print(f"Max dev {metric_label}: {max_dev_acc} at epoch {max_dev_acc_epoch}")


def min_max_fine_tune(options, model, data, params, np_random_state, bert_model, bert_device,
                      only_save_best_models=False):
    # This is intended for zero-shot sequence-level fine-tuning of a model that has already been initially trained
    # at the sentence-level, using, for example, train(). If --only_save_best_models, there's an option to either
    # save by the sentence-level prediction from the fully connected layer or
    # by the sentence-level accuracy where a positive prediction occurs if at least
    # one sequence label is positive. An alternative, not considered here, would be to choose the epoch based on a
    # small number of token-level annotated sentences. (Recall that in the standard zero-shot setting, we do not have
    # access to token-level labels; in this way, it reflects a standard sentence-level classification setting where
    # we would like to analyze token-level features but only have sentence-level labels.)

    print("Starting fine-tuning")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])

    seq_labeling_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    pre_dev_acc = 0
    max_dev_acc = 0
    max_dev_acc_epoch = -1

    num_batch_instances = math.ceil( (len(data["idx_train_x"]) / params["BATCH_SIZE"]) )
    all_epoch_cumulative_losses = []
    for e in range(params["EPOCH"]):
        if bert_model is not None:
            data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"], data["train_y"], \
            data["train_padded_seq_y"], data["train_padded_seq_y_mask"] = \
                shuffle(data["idx_train_x"], data["train_bert_idx_sentences"], data["train_bert_input_masks"],
                        data["train_y"], data["train_padded_seq_y"], data["train_padded_seq_y_mask"],
                        random_state=np_random_state)
        else:
            data["idx_train_x"], data["train_y"], data["train_padded_seq_y"], data["train_padded_seq_y_mask"] = \
                shuffle(data["idx_train_x"], data["train_y"], data["train_padded_seq_y"],
                        data["train_padded_seq_y_mask"], random_state=np_random_state)
        batch_num = 0
        cumulative_losses = []
        for i in range(0, len(data["idx_train_x"]), params["BATCH_SIZE"]):
            if batch_num % int(num_batch_instances * 0.25) == 0:
                print(f"Epoch {e+1}, {batch_num/num_batch_instances}")
                if len(cumulative_losses) > 0:
                    print(f"\tCurrent epoch average loss: {np.mean(cumulative_losses)}")
            batch_num += 1
            batch_range = min(params["BATCH_SIZE"], len(data["idx_train_x"]) - i)

            batch_x = data["idx_train_x"][i:i + batch_range]

            batch_train_padded_seq_y_mask = \
                torch.FloatTensor(data["train_padded_seq_y_mask"][i:i + batch_range]).to(params["main_device"])

            if bert_model is not None:
                # get BERT representations
                bert_output = \
                    utils_transformer.get_bert_representations(data["train_bert_idx_sentences"][i:i + batch_range],
                                                               data["train_bert_input_masks"][i:i + batch_range],
                                                               bert_model, bert_device, params["bert_layers"],
                                                               len(batch_x[0]))
            else:
                bert_output = None
            batch_y = data["train_y"][i:i + batch_range]

            if params["GPU"] != -1:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output))

            # construct min/max gold
            min_max_y = []
            for y in batch_y:
                if y == 0:
                    min_max_y.append([0, 0])
                elif y == 1:
                    min_max_y.append([0, 1])

            batch_min_max_y = torch.FloatTensor(min_max_y).to(params["main_device"])

            optimizer.zero_grad()
            model.train()

            pred_seq_labels = model(batch_x, bert_output,
                                    forward_type=constants.FORWARD_TYPE_SEQUENCE_LABELING,
                                    main_device=params["main_device"])

            if not options.do_not_allow_padding_to_participate_in_min_max_loss:
                # This was the version used in the original grammar work, so I keep this as the default. In this case,
                # a padding index/token could, in principle as an edge case, indirectly participate in the min-max
                # as a floor/ceiling if there's ever a sentence with
                # all positive or all negative token-level logits (which may often be rare, but the likelihood may have
                # an interaction/dependence/correlation with the overall sentence-level class balance),
                # as the padding indexes are always 0.
                # Whether or not this is preferable (or makes a difference) will depend on the dataset/setting.
                # The bias towards sparsity would seem to be stronger with the alternative (if the edge case
                # ever occurs), although with the grammar datasets, uniformly positive
                # contributions for class 1 are rare using the basic sentence-level loss, and this variation
                # works as anticipated, with the intended effect of increasing precision, but that assumption/behavior
                # may not be true for all datasets with varying class0/class1 balances,
                # so it may be worth considering the --do_not_allow_padding_to_participate_in_min_max_loss
                # version for other datasets. (That said, this version does seem to work well in practice.)
                # Also, note that we can replace the min/max with a top-k min/max if
                # applicable for a given dataset.
                contributions_min, _ = torch.min(batch_train_padded_seq_y_mask * pred_seq_labels, dim=1)
                contributions_max, _ = torch.max(batch_train_padded_seq_y_mask * pred_seq_labels, dim=1)
                min_max_token_contributions_tensor = torch.zeros(batch_x.shape[0], 2).to(params["main_device"])
                min_max_token_contributions_tensor[:, 0] = contributions_min
                min_max_token_contributions_tensor[:, 1] = contributions_max
            else:
                # In this alternative, we make the padding indexes very large/small so that those indexes are
                # never selected.
                contributions_min, _ = \
                    torch.min(((1 - batch_train_padded_seq_y_mask) * (10 ** 8)) + pred_seq_labels, dim=1)
                contributions_max, _ = \
                    torch.max(((1 - batch_train_padded_seq_y_mask) * (-10 ** 8)) + pred_seq_labels, dim=1)
                min_max_token_contributions_tensor = torch.zeros(batch_x.shape[0], 2).to(params["main_device"])
                min_max_token_contributions_tensor[:, 0] = contributions_min
                min_max_token_contributions_tensor[:, 1] = contributions_max

            loss = seq_labeling_criterion(min_max_token_contributions_tensor, batch_min_max_y)
            loss = loss.mean()

            cumulative_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        print(f"Epoch average loss: {np.mean(cumulative_losses)}")
        all_epoch_cumulative_losses.extend(cumulative_losses)
        print(f"Average loss across all mini-batches (all epochs): {np.mean(all_epoch_cumulative_losses)}")
        if options.eval_fine_tuning_epoch_based_on_local_to_global_accuracy:
            dev_acc, score_vals = utils_classification.test_based_on_contributions(data, model, params, bert_model,
                                                                                   bert_device, mode="dev")
        else:
            dev_acc, score_vals = utils_classification.test(data, model, params, bert_model, bert_device, mode="dev")
            global_acc_from_local, _ = utils_classification.test_based_on_contributions(data, model, params, bert_model,
                                                                                        bert_device, mode="dev")
            print(f"The sentence-level accuracy derived from the token-level logits: {global_acc_from_local}")

        print("epoch:", e + 1, "/ dev_acc:", dev_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc >= max_dev_acc:
            max_dev_acc = dev_acc
            max_dev_acc_epoch = e + 1
            if only_save_best_models:
                print(
                    f"Saving epoch {e + 1} as new best max_dev_acc_epoch model")
                utils.save_model_torch(model, params, f"max_dev_acc_epoch_epoch")

                print("Saving scores file")
                utils.save_lines(params["score_vals_file"] + f".epoch{e+1}.txt", score_vals)
                print(f"Saved scores file: {params['score_vals_file']}.epoch{e+1}.txt")

        print(f"\tCurrent max dev accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")

        # save after *every* epoch, unless only_save_best_models
        if e + 1 > 0:
            if params["SAVE_MODEL"] and not only_save_best_models:
                print(f"Saving epoch {e+1}")
                utils.save_model_torch(model, params, e + 1)
                print("Saving scores file")
                utils.save_lines(params["score_vals_file"] + f".epoch{e+1}.txt", score_vals)
                print(f"Saved scores file: {params['score_vals_file']}.epoch{e+1}.txt")

    print(f"Sentence-level max dev accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")
import utils
import constants
import utils_transformer
import utils_eval
import utils_viz
import utils_exemplar

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


def generate_exemplar_data(output_exemplar_data_file, data, model, params, bert_model, bert_device, mode="test", fce_eval=True):

    model.eval()

    output_exemplar_data_file_object = codecs.open(output_exemplar_data_file, 'w', 'utf-8')

    if mode == "dev":
        x, y, all_untruncated_tokenized_tokens, all_original_untokenized_tokens = \
            data["idx_dev_x"], data["dev_y"], data["dev_x"], data["dev_sentences"]
        if bert_model is not None:
            bert_idx_sentences, bert_input_masks, all_bert_to_original_tokenization_maps = \
                data["dev_bert_idx_sentences"], data["dev_bert_input_masks"], \
                data["dev_bert_to_original_tokenization_maps"]
        seq_y = data["dev_seq_y"]
    elif mode == "test":
        x, y, all_untruncated_tokenized_tokens, all_original_untokenized_tokens = \
            data["idx_test_x"], data["test_y"], data["test_x"], data["test_sentences"]
        if bert_model is not None:
            bert_idx_sentences, bert_input_masks, all_bert_to_original_tokenization_maps = \
                data["test_bert_idx_sentences"], data["test_bert_input_masks"], \
                data["test_bert_to_original_tokenization_maps"]
        seq_y = data["test_seq_y"]

    contribution_tuples_per_sentence = []
    sentence_probs = []

    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)

        batch_x = x[i:i + batch_range]

        if bert_model is not None:
            # get BERT representations
            bert_output = utils_transformer.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                  bert_input_masks[i:i + batch_range], bert_model, bert_device,
                                                  params["bert_layers"], len(batch_x[0]))
            bert_to_original_tokenization_maps = all_bert_to_original_tokenization_maps[i:i + batch_range]
        else:
            bert_output = None

        batch_y = y[i:i + batch_range]
        batch_seq_y = seq_y[i:i + batch_range]

        batch_x = torch.LongTensor(batch_x).to(params["main_device"])
        if bert_model is not None:
            bert_output = torch.FloatTensor(bert_output).to(params["main_device"])

        with torch.no_grad():
            uni_cnn_matrix_no_relu, token_contributions_tensor, output = \
                model(batch_x, bert_output, forward_type=constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS,
                      main_device=params["main_device"])

        uni_cnn_matrix_no_relu = uni_cnn_matrix_no_relu.cpu()
        output = output.cpu()
        token_contributions_tensor = token_contributions_tensor.cpu()

        if not fce_eval:
            assert False
            one_test_target_x = test_target_x[sentence_index]
            one_correction_generated_target_x = correction_generated_target_x[sentence_index]

        """
        The token_contributions_tensor contain all tokenized tokens in the input 
        (max_length+2*constants.PADDING_SIZE) (for the batch).
        The gold detection labels contain labels for each of the original tokens (which may be more than max_length).
        Here, we produce labels for each of the original tokens (up to max_length), ignoring padding symbols
        and setting any tokens beyond max_length to the default (i.e., correct/no-error label). In the case of BERT,
        we also need to detokenize (i.e., remove WordPieces) in order to restore alignment to the gold labels.
        """

        neg_logit_bias = model.fc.bias.cpu().data.numpy()[0]
        pos_logit_bias = model.fc.bias.cpu().data.numpy()[1]

        prob = F.softmax(output, dim=1)
        for sentence_index in range(token_contributions_tensor.shape[0]):

            unicnn_values = uni_cnn_matrix_no_relu[sentence_index,:,:]
            # print(f"unicnn_values: {unicnn_values}")
            # print(f"unicnn_values.shape: {unicnn_values.shape}") # torch.Size([1000, 58])

            neg_prob = prob[sentence_index][0].item()
            pos_prob = prob[sentence_index][1].item()
            sentence_probs.append([neg_prob, pos_prob])

            contribution_tuples = []
            for j in range(token_contributions_tensor.shape[1]):
                norm_val = token_contributions_tensor[sentence_index, j].item()
                # contribution_tuples.append((0.0, norm_val, neg_logit_bias, pos_logit_bias))
                contribution_tuples.append((0.0, norm_val, 0.0, 0.0))

            untruncated_tokenized_tokens = all_untruncated_tokenized_tokens[i:i + batch_range][sentence_index]
            # the original tokens (without additional tokenization) aligned with the gold labels
            original_untokenized_tokens = all_original_untokenized_tokens[i + sentence_index]

            # training_observed_length is the total number of real tokens seen by the model for this instance,
            # ignoring prefix/suffix padding:
            training_observed_length = min(params["max_length"], len(untruncated_tokenized_tokens))

            # fill remaining with default (in the case the original sentence has exceeded the truncated max length
            # used for training/eval):
            contribution_tuples = contribution_tuples[
                                  constants.PADDING_SIZE:constants.PADDING_SIZE + training_observed_length]
            contribution_tuples.extend([(0.0, 0.0, 0.0, 0.0)] * (
                len(untruncated_tokenized_tokens) - params["max_length"]))

            # analagous transformation for the uniCNN filter values:
            real_tokens_unicnn_values = \
                unicnn_values[:,constants.PADDING_SIZE:constants.PADDING_SIZE + training_observed_length]
            #print(f"real_tokens_unicnn_values.shape: {real_tokens_unicnn_values.shape}")
            # add 0's for any real tokens that were cut-off due to the max length restriction
            if real_tokens_unicnn_values.shape[1] < len(untruncated_tokenized_tokens):
                real_tokens_unicnn_values = torch.cat([real_tokens_unicnn_values,
                                                       torch.zeros(real_tokens_unicnn_values.shape[0],
                                                                   len(untruncated_tokenized_tokens) -
                                                                   real_tokens_unicnn_values.shape[1])]
                                                      ,dim=1)
            assert real_tokens_unicnn_values.shape[1] == len(contribution_tuples)
            ## END mask out padding

            if bert_model is not None:  # detokenize to match the original
                bert_to_original_tokenization_map = bert_to_original_tokenization_maps[sentence_index]
                assert len(bert_to_original_tokenization_map) == len(contribution_tuples)
                detokenized_generated_labels = defaultdict(list)  # keys are the original token indexes
                detokenized_real_tokens_unicnn_values = defaultdict(list)
                for tok_index, bert_to_token, gen_label in zip(range(len(bert_to_original_tokenization_map)),
                                                               bert_to_original_tokenization_map, contribution_tuples):
                    detokenized_generated_labels[bert_to_token].append(gen_label)
                    detokenized_real_tokens_unicnn_values[bert_to_token].append(real_tokens_unicnn_values[:,tok_index])

                generated_labels = []
                mean_detokenized_real_tokens_unicnn_values = []
                for original_token_id in range(len(detokenized_generated_labels)):
                    # Some original tokens were split by BERT, so the generated labels need to be combined:
                    neg_vals = []
                    pos_vals = []

                    for grad_tuple in detokenized_generated_labels[original_token_id]:
                        #neg_val, pos_val, neg_logit_bias, pos_logit_bias = grad_tuple
                        neg_val, pos_val, _, _ = grad_tuple
                        neg_vals.append(neg_val)
                        pos_vals.append(pos_val)
                    generated_labels.append((np.mean(neg_vals), np.mean(pos_vals), 0.0, 0.0))
                    #
                    if len(detokenized_real_tokens_unicnn_values[original_token_id]) > 1:
                        mean_detokenized_real_tokens_unicnn_values.append(
                            torch.mean(torch.stack(detokenized_real_tokens_unicnn_values[original_token_id]), dim=0).numpy())
                    else:
                        mean_detokenized_real_tokens_unicnn_values.append(
                            detokenized_real_tokens_unicnn_values[original_token_id][0].numpy())
                real_tokens_unicnn_values_by_word = mean_detokenized_real_tokens_unicnn_values
                contribution_tuples = generated_labels
            else:
                real_tokens_unicnn_values_by_word = []
                for tok_index in range(real_tokens_unicnn_values.shape[1]):
                    real_tokens_unicnn_values_by_word.append(real_tokens_unicnn_values[:,tok_index].numpy())
            assert len(contribution_tuples) == len(original_untokenized_tokens), \
                f"len(contribution_tuples): {len(contribution_tuples)}, " \
                f"len(original_untokenized_tokens): {len(original_untokenized_tokens)}, " \
                f"original_untokenized_tokens: {original_untokenized_tokens}"
            assert len(real_tokens_unicnn_values_by_word) == len(contribution_tuples)
            if not fce_eval:  # adding final holder sym
                assert False

            contribution_tuples_per_sentence.append(contribution_tuples)

            output_exemplar_data_file_object.write(f"SENT{i+sentence_index},{batch_y[sentence_index]},{neg_prob},{pos_prob},{neg_logit_bias},{pos_logit_bias}\n")
            for true_word_label, one_word_contribution, one_word_unicnn_values in zip(batch_seq_y[sentence_index],contribution_tuples,real_tokens_unicnn_values_by_word):
                _, norm_val, _, _ = one_word_contribution
                one_word_unicnn_values_as_string = [str(x) for x in list(one_word_unicnn_values)]

                output_exemplar_data_file_object.write(
                    f"{true_word_label},{norm_val},{','.join(one_word_unicnn_values_as_string)}\n")
            output_exemplar_data_file_object.flush()

    output_exemplar_data_file_object.close()

    return contribution_tuples_per_sentence, sentence_probs
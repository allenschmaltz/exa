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


def test_seq_labels_features(data, model, params, bert_model, bert_device, mode="test", fce_eval=True):
    # Note that sentence score aggregation is handled differently than the ngram aggregation. In particular,
    # note that duplicate sentences will have scores that are magnified (by each occurrence).
    # This may be desirable in some
    # cases (e.g., identifying common calls/slogans/etc.) and not in others (e.g., when duplicates are just a result
    # of data pre-processing).
    print(f"Note that the contribution_type in features is based only on the global fc layer, so this is primarily "
          f"intended to be used with the base sentence-level classifier decomposed to produce token-level scores. "
          f"(The original paper only considers that setting, for which it works rather well.) Some caution "
          f"is needed if this is used with a fine-tuned sequence-labeling model--either the fully supervised "
          f"setting or with the min-max fine-tuning, where the fc layer may have drifted from the original parameter "
          f"values, at least at a decision boundary of 0. "
          f"One possibility in those cases would be to "
          f"determine the sentence-level prediction via utils_classification.test_based_on_contributions().")
    model.eval()

    if mode == "dev":
        x, y, all_untruncated_tokenized_tokens, all_original_untokenized_tokens = data["idx_dev_x"], data["dev_y"], data[
            "dev_x"], data["dev_sentences"]
        if bert_model is not None:
            bert_idx_sentences, bert_input_masks, all_bert_to_original_tokenization_maps = \
            data["dev_bert_idx_sentences"], data["dev_bert_input_masks"], data["dev_bert_to_original_tokenization_maps"]
    elif mode == "test":
        x, y, all_untruncated_tokenized_tokens, all_original_untokenized_tokens = data["idx_test_x"], data["test_y"], data[
                "test_x"], data["test_sentences"]
        if bert_model is not None:
            bert_idx_sentences, bert_input_masks, all_bert_to_original_tokenization_maps = \
            data["test_bert_idx_sentences"], data["test_bert_input_masks"], data["test_bert_to_original_tokenization_maps"]

    # contribution_tuples_per_sentence = []
    # sentence_probs = []

    neg_unigrams = defaultdict(list)
    neg_bigrams = defaultdict(list)
    neg_trigrams = defaultdict(list)
    neg_4grams = defaultdict(list)
    neg_5grams = defaultdict(list)
    neg_sent_unnormalized = defaultdict(float)
    neg_sent_len_normalized = defaultdict(float)

    pos_unigrams = defaultdict(list)
    pos_bigrams = defaultdict(list)
    pos_trigrams = defaultdict(list)
    pos_4grams = defaultdict(list)
    pos_5grams = defaultdict(list)
    pos_sent_unnormalized = defaultdict(float)
    pos_sent_len_normalized = defaultdict(float)

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

        # batch_y = y[i:i + batch_range]

        batch_x = torch.LongTensor(batch_x).to(params["main_device"])
        if bert_model is not None:
            bert_output = torch.FloatTensor(bert_output).to(params["main_device"])

        with torch.no_grad():
            # token_contributions_tensor, output = model(batch_x, bert_output, forward_type=2,
            #                                            main_device=params["main_device"])
            token_contributions_tensor_neg, token_contributions_tensor_pos, output = \
                model(batch_x, bert_output, forward_type=constants.FORWARD_TYPE_FEATURE_EXTRACTION,
                      main_device=params["main_device"])

        output = output.cpu()
        token_contributions_tensor_neg = token_contributions_tensor_neg.cpu()
        token_contributions_tensor_pos = token_contributions_tensor_pos.cpu()

        if not fce_eval:
            assert False
            one_test_target_x = test_target_x[sentence_index]
            one_correction_generated_target_x = correction_generated_target_x[sentence_index]

        """
        The token_contributions_tensor contain all tokenized tokens in the input (max_length+2*constants.PADDING_SIZE) (for the batch).
        The gold detection labels contain labels for each of the original tokens (which may be more than max_length).
        Here, we produce labels for each of the original tokens (up to max_length), ignoring padding symbols
        and setting any tokens beyond max_length to the default (i.e., correct/no-error label). In the case of BERT,
        we also need to detokenize (i.e., remove WordPieces) in order to restore alignment to the gold labels.
        """

        # in this case, the bias is ignored, since the negative and positive contributions are treated separately
        # neg_logit_bias = model.fc.bias.cpu().data.numpy()[0]
        # pos_logit_bias = model.fc.bias.cpu().data.numpy()[1]

        prob = F.softmax(output, dim=1)
        # loop over negative and positive contributions
        for contribution_type, token_contributions_tensor in enumerate([token_contributions_tensor_neg, token_contributions_tensor_pos]):
            for sentence_index in range(token_contributions_tensor.shape[0]):

                neg_prob = prob[sentence_index][0].item()
                pos_prob = prob[sentence_index][1].item()
                #sentence_probs.append([neg_prob, pos_prob])

                contribution_tuples = []
                for j in range(token_contributions_tensor.shape[1]):
                    # norm_val = token_contributions_tensor[sentence_index, j].item()
                    # # contribution_tuples.append((0.0, norm_val, neg_logit_bias, pos_logit_bias))
                    # contribution_tuples.append((0.0, norm_val, 0.0, 0.0))

                    if contribution_type == 0:
                        if pos_prob <= neg_prob:
                            norm_val = token_contributions_tensor[sentence_index, j].item()
                            contribution_tuples.append(norm_val)
                        else:
                            contribution_tuples.append(0.0)
                    elif contribution_type == 1:
                        if pos_prob > neg_prob:
                            norm_val = token_contributions_tensor[sentence_index, j].item()
                            contribution_tuples.append(norm_val)
                        else:
                            contribution_tuples.append(0.0)

                untruncated_tokenized_tokens = all_untruncated_tokenized_tokens[i:i + batch_range][sentence_index]
                original_untokenized_tokens = all_original_untokenized_tokens[
                    i + sentence_index]  # the original tokens (without additional tokenization) aligned with the gold labels

                # training_observed_length is the total number of real tokens seen by the model for this instance, ignoring prefix/suffix padding:
                training_observed_length = min(params["max_length"], len(untruncated_tokenized_tokens))

                # fill remaining with default (in the case the original sentence has exceeded the truncated max length used for training/eval):
                contribution_tuples = contribution_tuples[
                                      constants.PADDING_SIZE:constants.PADDING_SIZE + training_observed_length]
                contribution_tuples.extend([0.0] * (
                len(untruncated_tokenized_tokens) - params["max_length"]))

                ## END mask out padding

                if bert_model is not None:  # detokenize to match the original
                    bert_to_original_tokenization_map = bert_to_original_tokenization_maps[sentence_index]
                    assert len(bert_to_original_tokenization_map) == len(contribution_tuples)
                    detokenized_generated_labels = defaultdict(list)  # keys are the original token indexes
                    for bert_to_token, gen_label in zip(bert_to_original_tokenization_map, contribution_tuples):
                        detokenized_generated_labels[bert_to_token].append(gen_label)

                    generated_labels = []
                    for original_token_id in range(len(detokenized_generated_labels)):
                        # Some original tokens were split by BERT, so the generated labels need to be combined:
                        norm_vals = []

                        for norm_val in detokenized_generated_labels[original_token_id]:
                            norm_vals.append(norm_val)
                        generated_labels.append(np.mean(norm_vals))
                        #
                    contribution_tuples = generated_labels
                assert len(contribution_tuples) == len(original_untokenized_tokens)


                # collect sentences and logits
                if contribution_type == 0:
                    neg_sent_unnormalized[" ".join(original_untokenized_tokens)] += np.sum(contribution_tuples)
                    neg_sent_len_normalized[" ".join(original_untokenized_tokens)] += np.mean(contribution_tuples)
                elif contribution_type == 1:
                    pos_sent_unnormalized[" ".join(original_untokenized_tokens)] += np.sum(contribution_tuples)
                    pos_sent_len_normalized[" ".join(original_untokenized_tokens)] += np.mean(contribution_tuples)

                # now, collect ngrams and logits
                for token_index in range(len(contribution_tuples)):
                    ngram = original_untokenized_tokens[token_index]
                    token_logit = contribution_tuples[token_index]
                    if contribution_type == 0:
                        neg_unigrams[ngram].append(token_logit)
                    elif contribution_type == 1:
                        pos_unigrams[ngram].append(token_logit)

                    if token_index < len(original_untokenized_tokens)-1:
                        ngram = " ".join(original_untokenized_tokens[token_index:token_index+2])
                        token_logit = np.sum(contribution_tuples[token_index:token_index+2])
                        if contribution_type == 0:
                            neg_bigrams[ngram].append(token_logit)
                        elif contribution_type == 1:
                            pos_bigrams[ngram].append(token_logit)

                    if token_index < len(original_untokenized_tokens) - 2:
                        ngram = " ".join(original_untokenized_tokens[token_index:token_index + 3])
                        token_logit = np.sum(contribution_tuples[token_index:token_index + 3])
                        if contribution_type == 0:
                            neg_trigrams[ngram].append(token_logit)
                        elif contribution_type == 1:
                            pos_trigrams[ngram].append(token_logit)

                    if token_index < len(original_untokenized_tokens) - 3:
                        ngram = " ".join(original_untokenized_tokens[token_index:token_index + 4])
                        token_logit = np.sum(contribution_tuples[token_index:token_index + 4])
                        if contribution_type == 0:
                            neg_4grams[ngram].append(token_logit)
                        elif contribution_type == 1:
                            pos_4grams[ngram].append(token_logit)

                    if token_index < len(original_untokenized_tokens) - 4:
                        ngram = " ".join(original_untokenized_tokens[token_index:token_index + 5])
                        token_logit = np.sum(contribution_tuples[token_index:token_index + 5])
                        if contribution_type == 0:
                            neg_5grams[ngram].append(token_logit)
                        elif contribution_type == 1:
                            pos_5grams[ngram].append(token_logit)

            if not fce_eval:  # adding final holder sym
                assert False

    # ngram lines
    neg_features_output_lines = []
    pos_features_output_lines = []

    neg_features_output_lines.append(f"Ngrams contributing to the negative class, sorted by logits\n")
    for dict_i, ngrams_dict in enumerate([neg_unigrams, neg_bigrams, neg_trigrams, neg_4grams, neg_5grams]):
        ngrams_sorted_by_val = sorted(ngrams_dict.items(), key=lambda kv: np.sum(kv[1]), reverse=True)
        neg_features_output_lines.append(f"####ngram size: {dict_i+1}; sorted by total logit\n")
        for ngram, token_logit in ngrams_sorted_by_val:
            neg_features_output_lines.append(f"{ngram}: {np.sum(token_logit)} || Occurrences: {len(token_logit)}\n")
        neg_features_output_lines.append(f"\n")

        ngrams_sorted_by_val = sorted(ngrams_dict.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
        neg_features_output_lines.append(f"####ngram size: {dict_i+1}; normalized by occurrence\n")
        for ngram, token_logit in ngrams_sorted_by_val:
            neg_features_output_lines.append(f"{ngram}: {np.mean(token_logit)} || Occurrences: {len(token_logit)}\n")
        neg_features_output_lines.append(f"\n")

    pos_features_output_lines.append(f"Ngrams contributing to the positive class, sorted by logits\n")
    for dict_i, ngrams_dict in enumerate([pos_unigrams, pos_bigrams, pos_trigrams, pos_4grams, pos_5grams]):
        ngrams_sorted_by_val = sorted(ngrams_dict.items(), key=lambda kv: np.sum(kv[1]), reverse=True)
        pos_features_output_lines.append(f"####ngram size: {dict_i+1}; sorted by total logit\n")
        for ngram, token_logit in ngrams_sorted_by_val:
            pos_features_output_lines.append(f"{ngram}: {np.sum(token_logit)} || Occurrences: {len(token_logit)}\n")
        pos_features_output_lines.append(f"\n")

        ngrams_sorted_by_val = sorted(ngrams_dict.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
        pos_features_output_lines.append(f"####ngram size: {dict_i+1}; normalized by occurrence\n")
        for ngram, token_logit in ngrams_sorted_by_val:
            pos_features_output_lines.append(f"{ngram}: {np.mean(token_logit)} || Occurrences: {len(token_logit)}\n")
        pos_features_output_lines.append(f"\n")

    # sentences
    neg_sentence_features_output_lines = []
    pos_sentence_features_output_lines = []

    sents_sorted_by_val = sorted(neg_sent_unnormalized.items(), key=lambda kv: kv[1], reverse=True)
    neg_sentence_features_output_lines.append(f"####Unnormalized scores (negative class)\n")
    for sent, token_logit in sents_sorted_by_val:
        neg_sentence_features_output_lines.append(f"{sent}: {token_logit}\n")
    neg_sentence_features_output_lines.append(f"\n")

    sents_sorted_by_val = sorted(neg_sent_len_normalized.items(), key=lambda kv: kv[1], reverse=True)
    neg_sentence_features_output_lines.append(f"####Mean (length normalized) scores (negative class)\n")
    for sent, token_logit in sents_sorted_by_val:
        neg_sentence_features_output_lines.append(f"{sent}: {token_logit}\n")


    sents_sorted_by_val = sorted(pos_sent_unnormalized.items(), key=lambda kv: kv[1], reverse=True)
    pos_sentence_features_output_lines.append(f"####Unnormalized scores (positive class)\n")
    for sent, token_logit in sents_sorted_by_val:
        pos_sentence_features_output_lines.append(f"{sent}: {token_logit}\n")
    pos_sentence_features_output_lines.append(f"\n")

    sents_sorted_by_val = sorted(pos_sent_len_normalized.items(), key=lambda kv: kv[1], reverse=True)
    pos_sentence_features_output_lines.append(f"####Mean (length normalized) scores (positive class)\n")
    for sent, token_logit in sents_sorted_by_val:
        pos_sentence_features_output_lines.append(f"{sent}: {token_logit}\n")

    return neg_features_output_lines, pos_features_output_lines, \
           neg_sentence_features_output_lines, pos_sentence_features_output_lines
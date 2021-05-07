import constants

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.total_length = kwargs["total_length"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["vocab_size"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.padding_idx = kwargs["padding_idx"]
        self.IN_CHANNEL = 1
        self.bert_emb_size = kwargs["bert_emb_size"]

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.embedding = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM, padding_idx=self.padding_idx)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.WORD_DIM+self.bert_emb_size) * self.FILTERS[i],
                             stride=self.WORD_DIM+self.bert_emb_size)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp, bert_output=None, forward_type=constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION,
                main_device=None):
        # assert forward_type in [constants.FORWARD_TYPE_FEATURE_EXTRACTION,
        #                         constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS,
        #                         constants.FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION,
        #                         constants.FORWARD_TYPE_SEQUENCE_LABELING,
        #                         constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION]

        if bert_output is not None:
            x = self.embedding(inp)
            # x.shape: batch size by total sequence length by inp dimension (i.e., word embedding size)
            x = torch.cat([x, bert_output], 2).view(-1, 1, (self.WORD_DIM+self.bert_emb_size) * self.total_length)
        else:
            x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.total_length)

        if forward_type == constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION:
            conv_results = [
                F.max_pool1d(F.relu(self.get_conv(i)(x)), self.total_length - self.FILTERS[i] + 1)
                    .view(-1, self.FILTER_NUM[i])
                for i in range(len(self.FILTERS))]

            x = torch.cat(conv_results, 1)
            x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
            x = self.fc(x)

            return x
        # constants.FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION was 0
        # constants.FORWARD_TYPE_SEQUENCE_LABELING was 1
        # constants.FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION was 2
        elif forward_type == constants.FORWARD_TYPE_SEQUENCE_LABELING or \
                forward_type == constants.FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION or \
                forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:  # seq + sentence-level
            # Calculate the convolutional decomposition. This is used for both zero-shot sequence labeling and
            # for fully supervised sequence labeling. For generating the exemplar vectors, we additionally return
            # the full tensor of filter applications.
            max_pool_outputs = []
            max_pool_outputs_indices = []
            for i in range(len(self.FILTERS)):
                max_pool, max_pool_indices = F.max_pool1d(F.relu(self.get_conv(i)(x)), self.total_length - self.FILTERS[i] + 1, return_indices=True)
                max_pool_outputs.append(max_pool.view(-1, self.FILTER_NUM[i]))
                max_pool_outputs_indices.append(max_pool_indices)

            concat_maxpool = torch.cat(max_pool_outputs, 1)

            concat_maxpool = F.dropout(concat_maxpool, p=self.DROPOUT_PROB, training=self.training)

            token_contributions_tensor = torch.zeros(concat_maxpool.shape[0], self.total_length).to(main_device)
            negative_contributions = concat_maxpool * self.fc.weight[0]
            positive_contributions = concat_maxpool * self.fc.weight[1]
            for maxpool_id in range(len(self.FILTERS)):
                for index_into_maxpool in range(self.FILTER_NUM[maxpool_id]):
                    contribution_id = sum(
                        self.FILTER_NUM[0:maxpool_id]) + index_into_maxpool

                    contributions = positive_contributions[:, contribution_id] - negative_contributions[:,
                                                                                 contribution_id]  # note that model.fc.bias[1] - model.fc.bias[0] gets added (once) after the loop

                    indexes_into_padded_sentence = max_pool_outputs_indices[maxpool_id][:, index_into_maxpool, 0]

                    for filter_index_offset in range(self.FILTERS[maxpool_id]):
                        index_into_padded_sentence = indexes_into_padded_sentence + filter_index_offset
                        token_contributions_tensor.scatter_add_(1, index_into_padded_sentence.view(-1, 1),
                                                                contributions.view(-1, 1))

            token_contributions_tensor.add_(self.fc.bias[1] - self.fc.bias[0])

            if forward_type == constants.FORWARD_TYPE_SEQUENCE_LABELING:
                return token_contributions_tensor
            elif forward_type == constants.FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION:
                return token_contributions_tensor, self.fc(concat_maxpool)
            elif forward_type == constants.FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS:
                assert len(self.FILTERS) == 1, f"ERROR: The exemplar setting only works with uniCNNs in this version."
                # Note that there is no relu on the exemplar vectors (a.k.a., uniCNN filter applications).
                # We also return the token contributions (for sequence labeling) and the sentence-level prediction,
                # as they get saved to the database data structures.
                return self.get_conv(0)(x), token_contributions_tensor, self.fc(concat_maxpool)
            else:
                assert False

        elif forward_type == constants.FORWARD_TYPE_FEATURE_EXTRACTION:  # for feature extraction
            # This is similar to the above, but here we separate the positive and negative class token-level
            # contributions.

            max_pool_outputs = []
            max_pool_outputs_indices = []
            for i in range(len(self.FILTERS)):
                max_pool, max_pool_indices = F.max_pool1d(F.relu(self.get_conv(i)(x)), self.total_length - self.FILTERS[i] + 1, return_indices=True)
                max_pool_outputs.append(max_pool.view(-1, self.FILTER_NUM[i]))
                max_pool_outputs_indices.append(max_pool_indices)

            concat_maxpool = torch.cat(max_pool_outputs, 1)

            concat_maxpool = F.dropout(concat_maxpool, p=self.DROPOUT_PROB, training=self.training)

            token_contributions_tensor_neg = torch.zeros(concat_maxpool.shape[0], self.total_length).to(main_device)
            token_contributions_tensor_pos = torch.zeros(concat_maxpool.shape[0], self.total_length).to(main_device)

            negative_contributions = concat_maxpool * self.fc.weight[0]
            positive_contributions = concat_maxpool * self.fc.weight[1]
            for maxpool_id in range(len(self.FILTERS)):
                for index_into_maxpool in range(self.FILTER_NUM[maxpool_id]):
                    contribution_id = sum(
                        self.FILTER_NUM[0:maxpool_id]) + index_into_maxpool

                    contributions_neg = negative_contributions[:, contribution_id]
                    contributions_pos = positive_contributions[:, contribution_id]

                    indexes_into_padded_sentence = max_pool_outputs_indices[maxpool_id][:, index_into_maxpool, 0]

                    for filter_index_offset in range(self.FILTERS[maxpool_id]):
                        index_into_padded_sentence = indexes_into_padded_sentence + filter_index_offset
                        token_contributions_tensor_neg.scatter_add_(1, index_into_padded_sentence.view(-1, 1),
                                                                    contributions_neg.view(-1, 1))
                        token_contributions_tensor_pos.scatter_add_(1, index_into_padded_sentence.view(-1, 1),
                                                                contributions_pos.view(-1, 1))

            #token_contributions_tensor.add_(self.fc.bias[1] - self.fc.bias[0])

            return token_contributions_tensor_neg, token_contributions_tensor_pos, self.fc(concat_maxpool)
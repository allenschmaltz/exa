import constants_exa

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LinearExA(nn.Module):
    def __init__(self, **kwargs):
        super(LinearExA, self).__init__()

        self.support_size = kwargs["support_size"]
        self.top_k = kwargs["top_k"]

        self.model_type = kwargs["model_type"]
        self.model_temperature_init_value = kwargs["model_temperature_init_value"]
        # [torch.linspace start, torch.linspace end, starting value for k==0]:
        self.model_support_weights_init_values = kwargs["model_support_weights_init_values"]
        self.model_bias_init_value = kwargs["model_bias_init_value"]
        self.model_gamma_init_value = kwargs["model_gamma_init_value"]

        self.model_bias = nn.Parameter(torch.tensor([self.model_bias_init_value]))  # default: 0.0
        self.model_bias.requires_grad = True

        self.model_gamma = nn.Parameter(torch.tensor([self.model_gamma_init_value]))  # default: 1.0
        self.model_gamma.requires_grad = True

        if self.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
            # initialize k support weights with decreasing weights; these are renormalized via a softmax in the forward,
            # but the loss min-max constraints operate on the logits in [-inf, inf] for the BCE (with logits) loss
            linspace_start = self.model_support_weights_init_values[0]
            linspace_end = self.model_support_weights_init_values[1]
            k1_start = self.model_support_weights_init_values[2]
            inital_model_support_weights = torch.linspace(linspace_start, linspace_end, self.top_k)  # 1.0,-1.0,2.0
            inital_model_support_weights[0] = k1_start  # for k=1
            self.model_support_weights = nn.Parameter(inital_model_support_weights)
            self.model_support_weights.requires_grad = True

        if self.model_type in [constants_exa.MODEL_KNN_MAXENT, constants_exa.MODEL_KNN_LEARNED_WEIGHTING]:
            self.model_temperature = nn.Parameter(torch.tensor([self.model_temperature_init_value]))
            self.model_temperature.requires_grad = True

        # These are the token contributions from the model over the support set
        self.model_prediction_logits = torch.zeros(self.support_size)
        self.model_prediction_logits.requires_grad = False  # Note that this is not a learnable parameter

        # These are the true labels from the model over the support set -- token-level labels in the fully supervised
        # setting, but otherwise (and typically) sentence/document level labels
        # In effect, this multiplied with gamma is a class-level bias offset, which is possible since these are from
        # the database over the support set
        self.model_true_labels = torch.zeros(self.support_size)
        self.model_true_labels.requires_grad = False  # Note that this is not a learnable parameter

    def update_model_prediction_logits(self, model_prediction_logits):
        self.model_prediction_logits = model_prediction_logits
        self.model_prediction_logits.requires_grad = False

    def update_model_true_labels(self, model_true_labels):
        self.model_true_labels = model_true_labels
        self.model_true_labels.requires_grad = False

    def softmax_with_temperature(self, vector_of_logits):
        # max-ent without the negative re-scaling
        renormed_logits = torch.exp(vector_of_logits / self.model_temperature)
        normed_denominator = torch.sum(renormed_logits, dim=0)
        renormed_logits = renormed_logits / normed_denominator
        return renormed_logits

    # def softmax_with_temperature_tensor(self, batch_tensor):
    #     # max-ent without the negative re-scaling
    #     renormed_tensor = torch.exp(batch_tensor / self.model_temperature)
    #     normed_denominator = torch.sum(renormed_tensor, dim=1)
    #     renormed_tensor = renormed_tensor / normed_denominator.unsqueeze(1)
    #     return renormed_tensor

    def maxent_with_temperature_tensor(self, batch_tensor):
        renormed_tensor = torch.exp(-1*batch_tensor / self.model_temperature)
        normed_denominator = torch.sum(renormed_tensor, dim=1)
        renormed_tensor = renormed_tensor / normed_denominator.unsqueeze(1)
        return renormed_tensor

    def forward(self, top_k_distances=None, top_k_distances_idx=None, pairwise_distances=None,
                return_nearest_distances=False, return_nearest_indexes=False, max_exemplars_to_return=1):

        if pairwise_distances is not None:
            assert top_k_distances is None and top_k_distances_idx is None
            top_k_distances, top_k_distances_idx = torch.topk(pairwise_distances, self.top_k, largest=False, sorted=True)
        else:
            # If we're given the top_k (from pre-caching), we don't need to call torch.topk again over the
            # full support set. TODO: Currently, this version only has this pre-caching setting implemented in K-NN
            # training and eval.
            assert pairwise_distances is None
            assert top_k_distances.shape == top_k_distances_idx.shape
            assert top_k_distances.shape[1] >= self.top_k, \
                f"ERROR: The cached distance structures must contain at least {self.top_k} sorted exemplar " \
                f"distances and indexes, but only {top_k_distances.shape[1]} were provided. " \
                f"Re-run the caching script with a larger --top_k."
            if top_k_distances.shape[1] > self.top_k:
                top_k_distances = top_k_distances[:, 0:self.top_k]
                top_k_distances_idx = top_k_distances_idx[:, 0:self.top_k]

        sorted_prediction_logits = torch.gather(self.model_prediction_logits.repeat(top_k_distances_idx.shape[0],1),
                                                dim=1, index=top_k_distances_idx)
        sorted_model_true_labels = torch.gather(self.model_true_labels.repeat(top_k_distances_idx.shape[0],1),
                                                dim=1, index=top_k_distances_idx)

        prediction_term = torch.tanh(sorted_prediction_logits)
        label_term = self.model_gamma * sorted_model_true_labels

        if self.model_type == constants_exa.MODEL_KNN_MAXENT:
            # self.maxent_with_temperature_tensor(top_k_distances) is a tensor of batch_size by K
            weighted_approximation = torch.sum(
                self.maxent_with_temperature_tensor(top_k_distances) * (prediction_term + label_term), dim=1)
        elif self.model_type == constants_exa.MODEL_KNN_BASIC:
            weighted_approximation = torch.mean(prediction_term + label_term, dim=1)
        elif self.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
            # self.softmax_with_temperature(self.model_support_weights) is a vector of length K
            weighted_approximation = torch.sum(
                self.softmax_with_temperature(self.model_support_weights) * (prediction_term + label_term), dim=1)

        if return_nearest_distances:  # inference
            # This only occurs at inference, so we do not need to return self.model_support_weights for
            # constants_exa.MODEL_KNN_LEARNED_WEIGHTING
            if return_nearest_indexes:
                return weighted_approximation + self.model_bias, top_k_distances[:, 0:max_exemplars_to_return], \
                       top_k_distances_idx[:, 0:max_exemplars_to_return]
            else:
                return weighted_approximation + self.model_bias, top_k_distances[:, 0:max_exemplars_to_return]
        else:  # training
            if self.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
                # we return the support weights without transformation
                return weighted_approximation + self.model_bias, self.model_support_weights
            else:
                return weighted_approximation + self.model_bias

    def get_model_term_decomposition_for_instance(self, top_k_distances_for_one_instance,
                                                  top_k_distances_idx_for_one_instance, as_string=True):
        # return a list of the model terms for a given vector of distances, which may not include the full K for some
        # model types (which is an option just to reduce the output file size, primarily just for debugging)
        # -1*bias is the new decision boundary
        k_under_consideration = top_k_distances_for_one_instance.shape[0]
        top_k_distances = top_k_distances_for_one_instance.unsqueeze(0)
        top_k_distances_idx = top_k_distances_idx_for_one_instance.unsqueeze(0)

        sorted_prediction_logits = torch.gather(self.model_prediction_logits.repeat(top_k_distances_idx.shape[0], 1),
                                                dim=1, index=top_k_distances_idx)
        sorted_model_true_labels = torch.gather(self.model_true_labels.repeat(top_k_distances_idx.shape[0], 1),
                                                dim=1, index=top_k_distances_idx)

        prediction_term = torch.tanh(sorted_prediction_logits)
        label_term = self.model_gamma * sorted_model_true_labels

        if self.model_type == constants_exa.MODEL_KNN_MAXENT:
            # TODO: Alternatively, we could just wait to make the truncation until right before saving the human
            # readable output, but typically this particular setting would only occur during debugging anyway and
            # probably should be altogether removed as an option to avoid confusion. (I.e., if this is used for
            # interpretability, we would require the full set of exemplars to always be shown to the end-user.)
            assert k_under_consideration == self.top_k, f"ERROR: The term decomposition is not valid in this case, " \
                                                        f"since the maxent calculation is over a different support " \
                                                        f"than the original model. Include the full top_k distances."
            model_support_weights = self.maxent_with_temperature_tensor(top_k_distances)[0]
        elif self.model_type == constants_exa.MODEL_KNN_BASIC:
            model_support_weights = torch.tensor([1/self.top_k]*self.top_k)  # for equally weighted mean
        elif self.model_type == constants_exa.MODEL_KNN_LEARNED_WEIGHTING:
            # softmax over all weights as in the original calculation, and then subsequently consider subsets, as
            # applicable (i.e., if k_under_consideration < self.top_k)
            model_support_weights = self.softmax_with_temperature(self.model_support_weights)

        # [sum( w_k * (tanh(f_n)+\gamma*y_n) )_{over all k}] + bias
        # n is the database index, which is the token index (not the sentence index), so it's really only useful for
        # debugging -- use database_token_to_sentence[nearest_database_index] to get the sentence index
        term_analysis = []
        for k, n, w_k, f_n, y, tanh_f_n, gamma_y_n in zip(range(top_k_distances_idx_for_one_instance.shape[0]),
                                                          top_k_distances_idx_for_one_instance,
                                                          model_support_weights[0:k_under_consideration],
                                                          sorted_prediction_logits[0], sorted_model_true_labels[0],
                                                          prediction_term[0], label_term[0]):  # [0] b/c .unsqueeze(0)

            exemplar_weight = (w_k*(tanh_f_n+gamma_y_n)).item()
            if as_string:
                term_analysis.append(
                    (f"k={k}", f"n={n.item()}", f"w_k={w_k.item()}",
                     f"tanh(f_n={f_n.item()})={tanh_f_n.item()}", f"gamma*(y_n={y.item()})={gamma_y_n.item()}",
                     f"w_k*(tanh_f_n+gamma*y_n)={exemplar_weight}"))
            else:
                term_analysis.append(exemplar_weight)
        if as_string:
            additional_model_info = [f"gamma: {self.model_gamma.item()}", f"bias: {self.model_bias.item()}",
                                     f"new decision boundary: {-1*self.model_bias.item()}"]
        else:
            additional_model_info = self.model_bias.item()
        return term_analysis, additional_model_info




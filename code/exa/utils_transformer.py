# TODO: note the conversion of the BERT output from numpy back to tensors should be dropped

import constants
import torch
import numpy as np


def get_bert_representations(train_bert_idx_sentences, train_bert_input_masks, bert_model,
                             bert_device, bert_layers, total_length):
    """
    Get BERT output representations and reshape them to match the padding of the CNN output (in preparation for concat)
    :param train_bert_idx_sentences:
    :param train_bert_input_masks:
    :param bert_model:
    :param bert_device:
    :param bert_layers:
    :param total_length:
    :return:
    """
    train_bert_idx_sentences = torch.tensor(train_bert_idx_sentences, dtype=torch.long).to(bert_device)
    train_bert_input_masks = torch.tensor(train_bert_input_masks, dtype=torch.long).to(bert_device)
    with torch.no_grad():
        all_encoder_layers, _ = bert_model(train_bert_idx_sentences, token_type_ids=None,
                                           attention_mask=train_bert_input_masks)
        top_layers = []
        for one_layer_index in bert_layers:
            top_layers.append(all_encoder_layers[one_layer_index])
        layer_output = torch.cat(top_layers, 2)

        # zero trailing padding
        masks = torch.unsqueeze(train_bert_input_masks, 2).float()
        layer_output = layer_output*masks

        bert_output_reshaped = torch.zeros((layer_output.shape[0], total_length, layer_output.shape[2]))
        bert_output_reshaped[:, constants.PADDING_SIZE-1:(constants.PADDING_SIZE-1)+layer_output.shape[1], :] = layer_output

    # this conversion to numpy() is probably unnecessary in the current version and is relatively expensive
    return bert_output_reshaped.detach().cpu().numpy()
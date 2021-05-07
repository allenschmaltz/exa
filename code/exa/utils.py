import constants

from sklearn.utils import shuffle

import codecs

from collections import defaultdict

import torch

import re  # used in fce_formatter


def fce_formatter(sentence):
    """
    Format sentence using the convention of earlier work on FCE grammar detection.
    This is not used for most of the experiments, for which we use cased input with the original
    digits, but 1 (but not all) of the earlier works used this convention, which we follow here for comparison purposes.
    :param sentence: List of tokens
    :return: List of tokens, each of which is lowercased, with digits replaced with 0
    """
    formatted_sentence = []
    for token in sentence:
        token = token.lower()
        token = re.sub(r'\d', '0', token)
        formatted_sentence.append(token)
    return formatted_sentence


def lowercase_formatter(sentence):
    """
    Format sentence by lowercasing every token. Most of our experiments use the cased BERT models, but some of the
    previous sentiment classification results used uncased models, for which this is used for comparison.
    :param sentence: List of tokens
    :return: List of tokens, each of which is lowercased
    """
    formatted_sentence = []
    for token in sentence:
        token = token.lower()
        formatted_sentence.append(token)
    return formatted_sentence


def fix_negations(sentence):
    """
    De-tokenize 'did n't'-style negations. This is to match the expected input format of the BERT tokenizer.
    :param sentence: List of tokens
    :return: De-tokenized list of tokens:
        Examples:
            did n't -> didn't
            can n't -> can't
    Assumes sentence does not start with "n't", which would suggest data problem. Note that when matching to
    ground-truth token-level labels, any such transformations need to be taken into account when de-tokenizing the
    WordPiece tokens if the labels correspond to the original 'did n't'-style format. See get_lines() for an
    example of how to handle such offsets.
    """
    formatted_sentence = []
    for token in sentence:
        if token == "n't":
            prev_token = formatted_sentence.pop()
            formatted_sentence.append(prev_token+"n't")
        else:
            formatted_sentence.append(token)
    return formatted_sentence


def get_lines(filepath_with_name, class_labels, data_formatter=None, tokenizer=None, input_is_untokenized=False):
    """
    Tokenize the input
    :param filepath_with_name: Filename string with: One sentence per line: classification label followed by one
    space and then the sentence: 0|1 Sentence text\n
    :param class_labels: List of int labels; labels in filepath_with_name must be members of this set
    :param data_formatter: None or "fce", which lowercases and replaces digits to 0, or "lowercase".
    This only exists to match the pre-processing conventions of some previous work
    :param tokenizer: None, or BertTokenizer; Converts input to WordPieces, if provided
    :return:
        sentences - List of tokenized sentences (each of which is a list of tokens)
        labels - List of int sentence labels
        bert_to_original_tokenization_maps - [] or if tokenizer is provided, list of list of ints mapping indices
            in tokenized sentences with index in original_sentences:
            e.g., len(bert_to_original_tokenization_maps[10]) == len(sentences[10])
        original_sentences: The untokenized sentences
    """
    labels = []
    sentences = []
    bert_to_original_tokenization_maps = []
    original_sentences = []  # without additional tokenization
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            assert constants.INS_START_SYM not in line and constants.INS_END_SYM not in line and constants.DEL_START_SYM not in line\
                   and constants.DEL_END_SYM not in line, "Diff symbols should not occur in the raw identification data."
            label = int(line[0])
            assert label in class_labels
            sentence = line[1:]
            original_sentence = list(sentence)
            if data_formatter and data_formatter != "":
                if data_formatter == "fce":
                    sentence = fce_formatter(sentence)
                elif data_formatter == "lowercase":
                    sentence = lowercase_formatter(sentence)
                else:
                    assert False, "ERROR: undefined data formatter"


            if tokenizer is not None:
                if not input_is_untokenized:
                    sentence = fix_negations(sentence)  # this is used to make the input match the expected input of the BERT tokenizer
                bert_to_original_tokenization_map = []  # maintain alignment for word-level labels
                token_i_neg_offset = 0
                bert_local_tokenization_check = []  # this is to check whether per-(whitespace-delimited-)word
                                                    # tokenization matches tokenization of the full sentence string
                                                    # (i.e., ensuring that running the tokenizer at the word level
                                                    # is the same as running at the sentence level)
                for token_i, token in enumerate(sentence):
                    bert_tokens = tokenizer.tokenize(token)
                    bert_local_tokenization_check.extend(bert_tokens)
                    for bert_token_i, _ in enumerate(bert_tokens):
                        if token.endswith("n't") and not input_is_untokenized:
                            if bert_token_i == 0:
                                bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                                token_i_neg_offset += 1  # original token was two tokens
                            else:
                                bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                        else:
                            bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                sentence = tokenizer.tokenize(" ".join(sentence))
                if len(bert_local_tokenization_check) != len(sentence) or " ".join(bert_local_tokenization_check) != " ".join(sentence):
                    assert False, F"ERROR: UNEXPECTED TOKENIZATION: per-token: {bert_local_tokenization_check} || sentence-level: {sentence}"
                bert_to_original_tokenization_maps.append(bert_to_original_tokenization_map)

            labels.append(label)
            sentences.append(sentence)
            original_sentences.append(original_sentence)
    return sentences, labels, bert_to_original_tokenization_maps, original_sentences


def convert_target_labels_lines_to_bert_tokenization(train_seq_y, train_bert_to_original_tokenization_maps):
    assert len(train_seq_y) == len(train_bert_to_original_tokenization_maps)
    converted_seq_y = []
    for sentence_id in range(len(train_seq_y)):
        one_seq_y = train_seq_y[sentence_id]
        one_bert_to_original_tokenization_maps = train_bert_to_original_tokenization_maps[sentence_id]
        new_seq_y = []
        assert len(one_seq_y) <= len(one_bert_to_original_tokenization_maps)
        for token_id in range(len(one_bert_to_original_tokenization_maps)):
            new_seq_y.append( one_seq_y[one_bert_to_original_tokenization_maps[token_id]] )
        converted_seq_y.append(new_seq_y)

    return converted_seq_y


def read_aesw_bert(training_file, dev_file, test_file, training_seq_labels_file=None, dev_seq_labels_file=None,
                   test_seq_labels_file=None, data_formatter=None, tokenizer=None, input_is_untokenized=False):
    """
    TODO: update these doc strings which are now stale
    Process the training, dev, and test files (text and sequence labels), as applicable (for train, test, etc.)
    :param training_file: Empty string, or filename string with: One sentence per line: classification label followed by one space and then the sentence: 0|1 Sentence text\n
    :param dev_file: Same format as training_file
    :param test_file: Same format as training_file
    :param training_seq_labels_file: None, or filename string with: One sentence per line: Must have 0 or 1 (separated by 1 space) for each token in training_file
    :param dev_seq_labels_file: None, or filename string with: One sentence per line: Must have 0 or 1 (separated by 1 space) for each token in dev_file
    :param test_seq_labels_file: None, or filename string with: One sentence per line: Must have 0 or 1 (separated by 1 space) for each token in test_file
    :param data_formatter: None or "fce", which lowercases and replaces digits to 0; Only exists to match some previous work
    :param tokenizer: None, or BertTokenizer; Used to convert input to WordPieces
    :return: A dictionary, whose contents varies depending on the provided arguments:
        if training_file != "":
            if training_seq_labels_file is None:
                data["train_x"], data["train_y"]: Shuffled list of tokenized sentences (no truncation) and list of int sentence labels
                data["dev_x"], data["dev_y"]: Non-shuffled list of tokenized sentences (no truncation) and list of int sentence labels
            else:
                data["train_x"], data["train_y"]: Shuffled list of tokenized sentences (no truncation) and list of int sentence labels
                data["dev_x"], data["dev_y"]: Non-shuffled list of tokenized sentences (no truncation) and list of int sentence labels

                data["train_seq_y"]: Shuffled (matching data["train_x"], data["train_y"]) list of token labels that match the tokenized sentences (including BERT, by spreading labels--see paper)
                data["dev_seq_y"]: Importantly, note that the dev token labels MATCH THE ORIGINAL, UNTOKENIZED SENTENCES
                data["dev_bert_to_original_tokenization_maps"]: [] or if tokenizer is provided, (list of) list of ints mapping indecies in sentences with index
                    in original_sentences; see get_lines(); Necessary to map data["dev_x"] to data["dev_seq_y"] (i.e., tokenized to original)
                data["dev_sentences"]: The original, untokenized sentences
        else:
            data["test_x"]: Non-shuffled list of tokenized sentences (no truncation)
            data["test_y"]: List of int sentence labels
            data["test_bert_to_original_tokenization_maps"]: Analogous to data["dev_bert_to_original_tokenization_maps"]
            data["test_sentences"]: The original, untokenized sentences
            data["test_seq_y"]: Importantly, note that the test token labels MATCH THE ORIGINAL, UNTOKENIZED SENTENCES
    """
    data = {}
    if training_file != "":
        if training_seq_labels_file is None:
            train_x, train_y, _, _ = get_lines(training_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            data["train_x"], data["train_y"] = shuffle(train_x, train_y, random_state=1776)
            data["dev_x"], data["dev_y"], _, _ = get_lines(dev_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
        else:
            assert dev_seq_labels_file is not None, "The dev sequence labels file must be provided, as well."
            train_x, train_y, train_bert_to_original_tokenization_maps, _ = get_lines(training_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            train_seq_y = get_target_labels_lines(training_seq_labels_file, convert_to_int=True)
            if tokenizer is not None:
                assert len(train_bert_to_original_tokenization_maps) > 0
                print(f"Converting/spreading training sequence labels to match Bert tokenization")
                train_seq_y = convert_target_labels_lines_to_bert_tokenization(train_seq_y, train_bert_to_original_tokenization_maps)

            data["train_x"], data["train_y"], data["train_seq_y"] = shuffle(train_x, train_y, train_seq_y, random_state=1776)
            data["dev_seq_y"] = get_target_labels_lines(dev_seq_labels_file, convert_to_int=True)
            data["dev_x"], data["dev_y"], data["dev_bert_to_original_tokenization_maps"], data["dev_sentences"] = get_lines(dev_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            if tokenizer is not None:
                data["dev_spread_seq_y"] = convert_target_labels_lines_to_bert_tokenization(data["dev_seq_y"],
                                                                               data["dev_bert_to_original_tokenization_maps"])
    elif test_file != "":
        data["test_x"], data["test_y"], data["test_bert_to_original_tokenization_maps"], data["test_sentences"] = get_lines(test_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
        if test_seq_labels_file is not None:
            data["test_seq_y"] = get_target_labels_lines(test_seq_labels_file, convert_to_int=True)
            if tokenizer is not None:
                data["test_spread_seq_y"] = convert_target_labels_lines_to_bert_tokenization(data["test_seq_y"],
                                                                               data["test_bert_to_original_tokenization_maps"])
    return data


def read_aesw_test_target_comparison_file(filepath_with_name):
    sentences = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            sentences.append(line)
    return sentences


def preprocess_sentences_seq_labels_for_training(seq_labels, max_length):
    padded_seq_y = []
    padded_seq_y_mask = []

    for labels in seq_labels:
        labels_truncated = labels[0:max_length]
        mask = [1] * len(labels_truncated)
        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        labels_truncated.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(labels_truncated)))
        mask.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(mask)))
        # add start padding
        labels_truncated = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + labels_truncated
        mask = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + mask
        assert len(labels_truncated) == max_length + 2*constants.PADDING_SIZE and len(labels_truncated) == len(mask)
        padded_seq_y.append(labels_truncated)
        padded_seq_y_mask.append(mask)
    return padded_seq_y, padded_seq_y_mask


def preprocess_sentences_without_bert(sentences, word_to_idx, max_length):
    idx_sentences = []
    for sentence in sentences:
        idx_sentence = []
        sentence_truncated = sentence[0:max_length]
        for token in sentence_truncated:
            if token in word_to_idx:
                idx_sentence.append(word_to_idx[token])
            else:
                idx_sentence.append(constants.UNK_SYM_ID)
        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        idx_sentence.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(idx_sentence)))
        # add start padding
        idx_sentence = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + idx_sentence
        assert len(idx_sentence) == max_length + 2*constants.PADDING_SIZE
        idx_sentences.append(idx_sentence)
    return idx_sentences


def preprocess_sentences(sentences, word_to_idx, max_length, tokenizer):
    if tokenizer is None:
        return preprocess_sentences_without_bert(sentences, word_to_idx, max_length), [], []

    # Note that the BERT [CLS] token is at the index of the last prefix padding for the CNN

    idx_sentences = []
    bert_idx_sentences = []
    bert_input_masks = []
    for sentence in sentences:
        idx_sentence = []
        bert_sentence = []
        bert_input_mask = []
        sentence_truncated = sentence[0:max_length]

        bert_sentence.append("[CLS]")
        bert_input_mask.append(1)
        for token in sentence_truncated:
            bert_sentence.append(token)
            bert_input_mask.append(1)
            if token in word_to_idx:
                idx_sentence.append(word_to_idx[token])
            else:
                idx_sentence.append(constants.UNK_SYM_ID)

        bert_sentence.append("[SEP]")
        bert_input_mask.append(1)

        bert_idx_sentence = tokenizer.convert_tokens_to_ids(bert_sentence)

        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        idx_sentence.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(idx_sentence)))
        # add start padding
        idx_sentence = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + idx_sentence

        # bert inputs are only padded at the end (to avoid complications with position embeddings with prefix padding)
        # The (max_length+2) is to account for [CLS] and [SEP]
        bert_idx_sentence.extend([0] * ((max_length+2) - len(bert_idx_sentence)))
        bert_input_mask.extend([0] * ((max_length+2) - len(bert_input_mask)))

        assert len(idx_sentence) == max_length + 2*constants.PADDING_SIZE
        assert len(idx_sentence) == len(bert_idx_sentence) + 2*constants.PADDING_SIZE - 2 and len(bert_idx_sentence) == len(bert_input_mask)
        idx_sentences.append(idx_sentence)
        bert_idx_sentences.append(bert_idx_sentence)
        bert_input_masks.append(bert_input_mask)

    return idx_sentences, bert_idx_sentences, bert_input_masks


def get_vocab(train_x, max_vocab_size):
    vocab = {}
    word_to_idx = {}
    idx_to_word = {}

    raw_vocab = defaultdict(int)
    for sentence in train_x:
        for word in sentence:
            raw_vocab[word] += 1

    print(f"Raw vocabulary size: {len(raw_vocab)}")
    # the max_vocab_size most frequent tokens, sorted:
    vocab_sorted_by_most_frequent = sorted(raw_vocab.items(), key=lambda kv: kv[1], reverse=True)[0:max_vocab_size]

    # the padding symbol and unk symbol must be the first two entries (in that order)
    assert [constants.PAD_SYM_ID, constants.UNK_SYM_ID] == [0, 1]

    for idx, word in enumerate([constants.PAD_SYM, constants.UNK_SYM]):
        vocab[word] = -1
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    for i, (word, freq) in enumerate(vocab_sorted_by_most_frequent):
        idx = i+2 # account for pad, unk
        vocab[word] = freq
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    print(f"Filtered vocabulary size: {len(vocab)} (including padding and unk symbols)")
    assert len(vocab) == len(word_to_idx) and len(word_to_idx) == len(idx_to_word) and len(idx_to_word) <= max_vocab_size + 2
    return vocab, word_to_idx, idx_to_word


def save_vocab(vocab_file, vocab, word_to_idx):
    """
    Save the vocabulary

    :param vocab_file: output filename at which to save the vocabulary
    :param vocab: word -> frequency (or -1 for special symbols)
    :param word_to_idx: word -> index
    :return: None; save file with the following format:
        token\t\index\tfrequency\n
    """
    vocab_lines = []
    vocab_sorted_by_idx = sorted(word_to_idx.items(), key=lambda kv: kv[1], reverse=False)
    for word, idx in vocab_sorted_by_idx:
        vocab_lines.append(f"{word}\t{idx}\t{vocab[word]}\n")

    save_lines(vocab_file, vocab_lines)
    print(f"Vocabulary saved to {vocab_file}")


def load_vocab(vocab_file):
    """
    Load vocabulary. See save_vocab() for the appropriate format.

    :param vocab_file: input filename from which to load the vocabulary
    :return: vocab, word_to_idx, idx_to_word
    """
    print(f"Loading existing vocabulary from {vocab_file}")
    vocab = {}
    word_to_idx = {}
    idx_to_word = {}
    with codecs.open(vocab_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 3
            word = line[0]
            idx = int(line[1])
            freq = int(line[2])
            vocab[word] = freq
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    assert word_to_idx[constants.PAD_SYM] == constants.PAD_SYM_ID and word_to_idx[constants.UNK_SYM] == constants.UNK_SYM_ID
    print(f"Loaded vocabulary size: {len(vocab)} (including padding and unk symbols)")
    return vocab, word_to_idx, idx_to_word


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def save_model_torch(model, params, epoch):
    path = f"{params['save_dir']}/{params['DATASET']}_{params['MODEL']}_{epoch}.pt"
    torch.save(model, path)
    print(f"A model was saved successfully as {path}")


def load_model_torch(filename, onto_gpu_id):

    try:
        if onto_gpu_id != -1:
            model = torch.load(filename, map_location=lambda storage, loc: storage.cuda(onto_gpu_id))
        else:
            model = torch.load(filename, map_location=lambda storage, loc: storage)
        print(f"Model in {filename} loaded successfully!")

        return model
    except:
        print(f"No available model such as {filename}.")


################################ for zero shot:

def get_target_labels_lines(filepath_with_name, convert_to_int=True):
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if convert_to_int:
                line = [int(x) for x in line]
            lines.append(line)
    return lines
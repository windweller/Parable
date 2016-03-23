"""
trim down GoogleNews-vector
based on the words in SNLI
"""

import os, json
import numpy as np
import unicodedata
import re

label_idx_map = {'-': 0, 'entailment': 1, 'neutral': 2, 'contradiction': 3}

word_idx_map = {'<NULL>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}

idx_word_map = ['<NULL>', '<UNK>', '<START>', '<END>']

word_count_map = {}


def load_dataset(base_dir, generate_word_dict):
    data = {}

    train_file = os.path.join(base_dir, 'snli_1.0_train.jsonl')

    get_data('train', train_file, data, generate_word_dict)

    dev_file = os.path.join(base_dir, 'snli_1.0_dev.jsonl')

    get_data('dev', dev_file, data, generate_word_dict)

    test_file = os.path.join(base_dir, 'snli_1.0_test.jsonl')

    get_data('test', test_file, data, generate_word_dict)

    return data


def get_data(category, file_path, data, generate_word_dict=False):
    """
    Args:
        category: 'train', 'dev', or 'test'
        data: pass in the dictionary, and we fill it up inside this function
    """
    data[category + '_sentences'] = []
    data['y_' + category] = []

    with open(file_path, 'r') as f:
        for line in f:
            json_obj = json._default_decoder.decode(line)
            if json_obj['gold_label'] == '-':  # skipping non-label
                continue
            pair = {}
            pair['pairID'] = json_obj['pairID']
            pair['sentence1'] = clean_str(json_obj['sentence1'])
            pair['sentence2'] = clean_str(json_obj['sentence2'])
            if generate_word_dict:
                sentence_array = pair['sentence1'].split()
                for word in sentence_array:
                    if word not in word_idx_map:
                        word_idx_map[word] = len(word_idx_map)
            # pair['sentence1_parse'] = json_obj['sentence1_parse']
            # pair['sentence2_parse'] = json_obj['sentence2_parse']

            data['y_' + category].append(int(label_idx_map[json_obj['gold_label']]))
            data[category + '_sentences'].append(pair)

    data['y_' + category] = np.asarray(data['y_' + category], dtype='int32')


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    data = load_dataset(pwd + "/snli_1.0", generate_word_dict=True)

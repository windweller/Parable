"""
trim down GoogleNews-vector
based on the words in SNLI
"""

import os, json
import numpy as np
import unicodedata
import re
from gensim.models.word2vec import Word2Vec
import time

label_idx_map = {'-': 0, 'entailment': 1, 'neutral': 2, 'contradiction': 3}

word_idx_map = {'<NULL>': 0, '<UNK>': 1, '<END>': 2}

idx_word_map = ['<NULL>', '<UNK>', '<END>']

# we replace rare words with <UNK>, which shares the same vector
word_count_map = {}  # length: 34044

max_seq_len = 0

W_embed = None


def load_dataset(base_dir):
    data = {}

    train_file = os.path.join(base_dir, 'snli_1.0_train.jsonl')

    get_data('train', train_file, data)

    dev_file = os.path.join(base_dir, 'snli_1.0_dev.jsonl')

    get_data('dev', dev_file, data)

    test_file = os.path.join(base_dir, 'snli_1.0_test.jsonl')

    get_data('test', test_file, data)

    return data


def get_data(category, file_path, data):
    """
    Args:
        category: 'train', 'dev', or 'test'
        data: pass in the dictionary, and we fill it up inside this function
    """
    global max_seq_len

    data[category + '_src_sentences'] = []
    data[category + '_tgt_sentences'] = []
    data['y_' + category] = []

    with open(file_path, 'r') as f:
        for line in f:
            json_obj = json._default_decoder.decode(line)
            if json_obj['gold_label'] == '-':  # skipping non-label
                continue
            sentence1 = clean_str(json_obj['sentence1'])
            sentence2 = clean_str(json_obj['sentence2'])

            sentence1_array = sentence1.split()
            sentence2_array = sentence2.split()

            for word in sentence1_array:
                if word not in word_idx_map:
                    word_idx_map[word] = len(word_idx_map)
                    idx_word_map.append(word)
                if word not in word_count_map:
                    word_count_map[word] = 0
                else:
                    word_count_map[word] += 1

            for word in sentence2_array:
                if word not in word_idx_map:
                    word_idx_map[word] = len(word_idx_map)
                    idx_word_map.append(word)
                if word not in word_count_map:
                    word_count_map[word] = 0
                else:
                    word_count_map[word] += 1

            if len(sentence1_array) > max_seq_len:
                max_seq_len = len(sentence1_array)
            elif len(sentence2_array) > max_seq_len:
                max_seq_len = len(sentence2_array)

            data['y_' + category].append(int(label_idx_map[json_obj['gold_label']]))
            data[category + '_src_sentences'].append(sentence1)
            data[category + '_tgt_sentences'].append(sentence2)

    data['y_' + category] = np.asarray(data['y_' + category], dtype='int32')


def decode(unicode_str):
    return unicodedata.normalize('NFKD', unicode_str).encode('ascii', 'ignore')


def convert_words_to_idx(data_X):
    """
    We convert word sentence into idx sentence,
    and if a word is not in word2vec: "rare", we already have a randomized word embedding

    Args:
        data_X: the 'train_sentences', 'dev_sentences', 'test_sentences'

    Returns:
    """

    converted = np.zeros((len(data_X), max_seq_len), dtype='int32')

    for i, sen in enumerate(data_X):
        sen_idx = np.zeros(max_seq_len, dtype='int32')

        for j, word in enumerate(decode(sen).split()):
            sen_idx[j] = word_idx_map[word]

        sen_idx[len(sen)] = word_idx_map['<END>']  # append <END> token to it

        converted[i, :] = sen_idx

    return converted


def compress_word2vec(W_embed, model):
    """
    We compress word2vec's 1.5G file with
    only the words we have

    update W_embed

    word2vec: the word2vec model we loaded

    Returns:
    """

    num_words_not_in = 0

    for i, word in enumerate(idx_word_map):
        if word in model:
            W_embed[i, :] = model[word]
        else:
            num_words_not_in += 1

    print "words not in word2vec: ", num_words_not_in


def print_stats(threshold=1, display=20):
    """
    print out how many words are equal to or below threshold

    In SNLI we have 12534 rare words (that appeared only once) (almost 33% of the corpus)

    display: how many of such words we want to display
    """
    rare_words = []

    for k, v in word_count_map.iteritems():
        if v <= threshold:
            rare_words.append(k)

    print "total number of rare words are: ", len(rare_words)
    # words not in word2vec:  6636

    for i in xrange(display):
        print rare_words[i]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC

    (this removes "." period as well)
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
    begin = time.time()

    pwd = os.path.dirname(os.path.realpath(__file__))

    data = load_dataset(pwd + "/snli_1.0")

    max_seq_len += 1

    data['train_src_sentences'] = convert_words_to_idx(data['train_src_sentences'])
    data['dev_src_sentences'] = convert_words_to_idx(data['dev_src_sentences'])
    data['test_src_sentences'] = convert_words_to_idx(data['test_src_sentences'])

    data['train_tgt_sentences'] = convert_words_to_idx(data['train_tgt_sentences'])
    data['dev_tgt_sentences'] = convert_words_to_idx(data['dev_tgt_sentences'])
    data['test_tgt_sentences'] = convert_words_to_idx(data['test_tgt_sentences'])

    print "data loaded..."

    model = Word2Vec.load_word2vec_format(pwd + '/GoogleNews-vectors-negative300.bin.gz', binary=True)

    print "word2vec loaded..."

    # plan: we map words that appears less than 5 times to a special token <UNK>
    # words that are frequent, more than 5 times, and not in word2vec,
    # we add a random vector to it.

    # initialize all embeddings randomly, then we swap out
    # words that appear in Word2Vec

    W_embed = np.random.randn(len(idx_word_map), 300)

    W_embed /= 100

    compress_word2vec(W_embed, model)

    with open(pwd + '/snli_vocab.json', 'w') as outfile:
        json.dump({idx_word_map: idx_word_map, word_idx_map: word_idx_map}, outfile)

    np.savez_compressed(pwd + "/snli_processed", W_embed=W_embed,
                        train_src_sentences=data['train_src_sentences'],
                        dev_src_sentences=data['dev_src_sentences'],
                        test_src_sentences=data['test_src_sentences'],
                        train_tgt_sentences=data['train_tgt_sentences'],
                        dev_tgt_sentences=data['dev_tgt_sentences'],
                        test_tgt_sentences=data['test_tgt_sentences'],
                        y_train=data['y_train'],
                        y_dev=data['y_dev'],
                        y_test=data['y_test'])

    end = time.time()

    print "time spent: ", (end - begin), "s"
    # 271.000173807 s

# coding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import json
import tarfile
import ConfigParser

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import math, os, sys
reload(sys)
sys.setdefaultencoding('utf8')
config = ConfigParser.ConfigParser()
config.read('config')
# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_GO1 = b"_GO1"
_GO2 = b"_GO2"
_GO3 = b"_GO3"
_GO4 = b"_GO4"
_GO5 = b"_GO5"
_GO6 = b"_GO6"
_GO7 = b"_GO7"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _UNK, _EOS, _GO]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
GO_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")



def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data, max_vocabulary_size,
                                            tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data" % (vocabulary_path))
        print(len(data))
        vocab = {}
        counter = 0
        num = 0
        for line in data:
            counter += 1
            if counter % 100000 == 0:
                print("    processing line %d" % counter)
            line = tf.compat.as_bytes(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
                num += 1
                word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
            overlap = .0
            for key in vocab_list[len(_START_VOCAB):]:
                overlap += vocab[key]
            print("overlap %f" % (overlap / num))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().decode('utf8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data, post_vocabulary_path, response_vocabulary_path,
                                            tokenizer=None, normalize_digits=True):
    print("Tokenizing data")
    post_vocab, _ = initialize_vocabulary(post_vocabulary_path)
    response_vocab, _ = initialize_vocabulary(response_vocabulary_path)
    counter = 0
    for pair in data:
        pair[0][0] = sentence_to_token_ids(pair[0][0], post_vocab, tokenizer, normalize_digits)
        for response in pair[1]:
            counter += 1
            if counter % 100000 == 0:
                print("    tokenizing pair %d" % counter)
            response[0] = sentence_to_token_ids(response[0], response_vocab, tokenizer, normalize_digits)


def prepare_data(data_dir, post_vocabulary_size, response_vocabulary_size, tokenizer=None):

    # Get data to the specified directory.
    train_path = os.path.join(data_dir, config.get('data', 'raw_train_file'))
    dev_path = os.path.join(data_dir, config.get('data', 'raw_dev_file'))

    tokenids_train_path = os.path.join(data_dir, config.get('data', 'train_file'))
    tokenids_dev_path = os.path.join(data_dir, config.get('data', 'dev_file'))
    tokenids_test_path = os.path.join(data_dir, config.get('data', 'test_file'))

    response_vocab_path = os.path.join(data_dir, config.get('data', 'response_vocab_file') % response_vocabulary_size)
    post_vocab_path = os.path.join(data_dir, config.get('data', 'post_vocab_file') % post_vocabulary_size)

    if not gfile.Exists(tokenids_train_path) or not gfile.Exists(tokenids_dev_path):

        train = json.load(open(train_path,'r'))
        dev = json.load(open(dev_path,'r'))
        # Create vocabularies of the appropriate sizes.
        create_vocabulary(response_vocab_path, [x[0][0] for x in train], response_vocabulary_size, tokenizer)
        create_vocabulary(post_vocab_path, [y[0] for x in train for y in x[1]], post_vocabulary_size, tokenizer)

        # Create token ids for the training data.
        data_to_token_ids(train, post_vocab_path, response_vocab_path, tokenizer)

        # Create token ids for the development data.
        data_to_token_ids(dev, post_vocab_path, response_vocab_path, tokenizer)

        # Write data
        with open(tokenids_train_path, 'w') as output:
            output.write(json.dumps(train, encoding='utf8', ensure_ascii=False))
        with open(tokenids_dev_path, 'w') as output:
            output.write(json.dumps(dev, encoding='utf8', ensure_ascii=False))

    return (tokenids_train_path, tokenids_dev_path, tokenids_test_path, post_vocab_path, response_vocab_path)    

def load_word_vector(fname):
    dic = {}
    with open(fname) as f:
        data = f.readlines()
        for line in data:
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            dic[word] = vector
    return dic

def load_vocab(fname):
    vocab = []
    with open(fname) as f:
        data = f.readlines()
        for d in data:
            vocab.append(d[:-1])
    return vocab

def random_init(dim):
    return 2 * math.sqrt(3) * (np.random.rand(dim) - 0.5) / math.sqrt(dim)

def refine_wordvec(rvector, vocab, dim=100):
    wordvec = []
    count = 0
    found = 0
    for word in vocab:
        count += 1
        if word in rvector:
            found += 1
            wordvec.append(np.array(list(map(float, rvector[word].split()))))
        else:
            wordvec.append(np.array(random_init(dim)))
    print('Total words: %d, Found words: %d, Overlap: %f' % (count, found, float(found)/count))
    return np.array(wordvec)

def get_data(data_dir, post_vocabulary_size, response_vocabulary_size):
    import scipy.io
    path = os.path.join(data_dir, config.get('data', 'wordvec'))
    try:
        mdict = scipy.io.loadmat(path)
        wordvec_post = mdict['post']
        wordvec_response = mdict['response']
    except:
        print('loading word vector...')
        raw_vector = load_word_vector(config.get('data', 'raw_wordvec'))
        print('loading vocabulary...')
        vocab_post = load_vocab(os.path.join(data_dir, config.get('data', 'post_vocab_file') % post_vocabulary_size))
        vocab_response = load_vocab(os.path.join(data_dir, config.get('data', 'response_vocab_file') % response_vocabulary_size))
        print('refine word vector...')
        wordvec_post = refine_wordvec(raw_vector, vocab_post)
        wordvec_response = refine_wordvec(raw_vector, vocab_response)
        mdict = {'post': wordvec_post, 'response': wordvec_response}
        scipy.io.savemat(path, mdict=mdict)
            
    return wordvec_post, wordvec_response

def get_ememory(data_dir, response_vocabulary_size):
    dic_path = os.path.join(data_dir, config.get('data', 'ememory_vocab_file') % response_vocabulary_size)
    vocab_response, _ = initialize_vocabulary(os.path.join(data_dir, config.get('data', 'response_vocab_file') % response_vocabulary_size))
    dic = json.load(open(dic_path, 'r'))
    emem = []
    for i in xrange(6):
        #if i == 0:
        #    emem.append(np.ones(response_vocabulary_size, dtype='float32'))
        #else:
        vec = [0] * response_vocabulary_size
        for j in dic[i]:
        #    print(j, vocab_response[j])
            if j in vocab_response:
                vec[vocab_response[j]] = 1
        emem.append(np.array(vec, dtype='float32'))
    emem = np.array(emem, dtype='float32')
    return emem
    


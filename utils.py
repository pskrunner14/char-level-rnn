import logging

import torch
import torch.nn as nn
import numpy as np

SOS_TOKEN = '~'
PAD_TOKEN = '#'
EOS_TOKEN = '^'

def to_matrix(sequences, token_to_idx):
    """Casts a list of names into rnn-digestable padded tensor"""
    seq_idx = []
    for seq in sequences:
        seq_idx.append([token_to_idx[token] for token in seq])
    sequences = [torch.LongTensor(x) for x in seq_idx]
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=token_to_idx[PAD_TOKEN])

def load_dataset(file_name='data/names'):

    with open(file_name) as file:
        names = file.read()[:-1].split('\n')
        names = [SOS_TOKEN + name.lower() for name in names]

    logging.info('number of sequences: {}'.format(len(names)))
    for name in names[::1000]:
        print(name[1:].capitalize())

    MAX_LENGTH = max(map(len, names))
    logging.info('max length: {}'.format(MAX_LENGTH))

    tokens = list(set([token for name in names for token in name]))
    tokens.append(PAD_TOKEN)
    n_tokens = len(tokens)
    logging.info('number of unique tokens: {}'.format(n_tokens))

    token_to_idx = {token: tokens.index(token) for token in tokens}
    assert len(tokens) ==  len(token_to_idx), 'dicts must have same lenghts'

    logging.debug('processing tokens')
    names = to_matrix(names, token_to_idx)
    return names, token_to_idx, tokens

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(inputs.size(0))
    for start_idx in range(0, inputs.size(0) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]
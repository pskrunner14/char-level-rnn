import os
import click
import logging
import coloredlogs

import torch
import torch.nn as nn

from utils import load_dataset
from model import CharRNN

SOS_TOKEN = '~'
PAD_TOKEN = '#'

@click.command()
@click.option(
    '-f',
    '--filename',
    default='data/names',
    type=click.Path(exists=True),
    help='path for the training data file'
)
@click.option(
    '-es',
    '--emb-size',
    default=32,
    help='size of the each embedding'
)
@click.option(
    '-hs',
    '--hidden-size',
    default=128,
    help='number of hidden RNN units'
)
def train(filename, emb_size, hidden_size):
    logging.info('reading `{}` for character sequences'.format(filename))
    seq_tensors, n_tokens = load_dataset(file_name=filename)
    
    logging.debug('creating char-level RNN model')
    model = CharRNN(n_tokens=n_tokens, emb_size=emb_size, hidden_size=hidden_size, pad_id=n_tokens - 1)



def main():
    coloredlogs.install(level='DEBUG')
    try:
        train()
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()
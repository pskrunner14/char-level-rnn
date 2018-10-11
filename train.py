import os
import click
import logging
import coloredlogs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from utils import load_dataset, iterate_minibatches
from model import CharRNN

SOS_TOKEN = '~'
PAD_TOKEN = '#'
EOS_TOKEN = '^'

@click.command()
@click.option(
    '-f', '--filename', default='data/names',
    type=click.Path(exists=True), help='path for the training data file'
)
@click.option(
    '-es', '--emb-size', default=32,
    help='size of the each embedding'
)
@click.option(
    '-hs', '--hidden-size', default=128,
    help='number of hidden RNN units'
)
@click.option(
    '-n', '--num-epochs', default=50,
    help='number of epochs for training'
)
@click.option(
    '-bz', '--batch-size', default=16,
    help='number of samples per mini-batch'
)
@click.option(
    '-lr', '--learning-rate', default=0.0001,
    help='learning rate for the adam optimizer'
)
@click.option(
    '-se', '--save-every', default=10,
    help='epoch interval for saving the model'
)
@click.option(
    '-sa', '--sample-every', default=5,
    help='epoch interval for sampling new sequences'
)
def train(filename, emb_size, hidden_size, num_epochs, batch_size, learning_rate, save_every, sample_every):
    """ Trains a character-level Recurrent Neural Network in PyTorch.

    Args: optional arguments [python train.py --help]
    """
    logging.info('reading `{}` for character sequences'.format(filename))
    inputs, token_to_idx, idx_to_token = load_dataset(file_name=filename)
    
    n_tokens = len(idx_to_token)
    max_length = inputs.size(1)
    
    logging.debug('creating char-level RNN model')
    model = CharRNN(n_tokens=n_tokens, emb_size=emb_size, 
                    hidden_size=hidden_size, pad_id=token_to_idx[PAD_TOKEN])
    inputs.cuda()
    model.cuda()
    
    criterion = nn.NLLLoss(reduction='elementwise_mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, 
                                                    factor=0.1, patience=7, verbose=True)

    split_index = int(0.9 * inputs.size(0))
    train_tensors, tensors = inputs[: split_index], inputs[split_index: ]
    split_index = int(0.5 * tensors.size(0))
    val_tensors, test_tensors = tensors[: split_index], tensors[split_index: ]
    del inputs, tensors
    logging.info('train tensors: {}'.format(train_tensors.size()))
    logging.info('val tensors: {}'.format(val_tensors.size()))
    logging.info('test tensors: {}'.format(test_tensors.size()))
    
    logging.debug('training char-level RNN model')
    # loop over epochs
    for epoch in range(1, num_epochs + 1):
        n_iter = 0
        epoch_loss = 0.0
        # loop over batches
        for batch in tqdm(iterate_minibatches(train_tensors, batchsize=batch_size, shuffle=True),
                        desc='Epoch[{}/{}]'.format(epoch, num_epochs), leave=False, 
                        total=train_tensors.size(0) // batch_size):
            outputs = torch.Tensor(max_length, batch_size, n_tokens)
            # loop over time-steps
            for t in range(max_length):
                input_tensor = batch[:, t]
                outputs[t] = model(input_tensor.cuda())
            targets = batch[:, 1: ].contiguous().view(-1)
            epoch_loss += optimize(model, outputs, targets, n_tokens, criterion, optimizer)
            n_iter += 1

        outputs = torch.Tensor(max_length, val_tensors.size(0), n_tokens)
        for t in range(max_length):
            input_tensor = val_tensors[:, t]
            outputs[t] = model(input_tensor.cuda())
        targets = val_tensors[:, 1: ].contiguous().view(-1)
        val_loss = forward(model, outputs, targets, n_tokens, criterion).item()
        scheduler.step(val_loss)

        logging.info('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'
                    .format(epoch, num_epochs, epoch_loss / n_iter, val_loss))
        
        # sample from the model every few epochs
        if epoch % sample_every == 0:
            for _ in range(10):
                sample = generate_sample(model, token_to_idx, idx_to_token, max_length, seed_phrase='~')
                logging.debug(sample)

def forward(model, outputs, targets, n_tokens, criterion):
    outputs = outputs.permute(1, 0, 2)
    outputs = outputs[:, :-1, :]
    outputs = outputs.contiguous().view(-1, n_tokens)
    loss = criterion(outputs, targets)
    return loss

def optimize(model, outputs, targets, n_tokens, criterion, optimizer):
    loss = forward(model, outputs, targets, n_tokens, criterion)
    loss.backward()
    optimizer.step()
    return loss.item()

def generate_sample(model, token_to_idx, idx_to_token, max_length, seed_phrase=SOS_TOKEN):
    """ Generates samples using seed phrase.
    
    This function generates text given a `seed_phrase` as initial states. 
    Remember to include start_token in seed phrase.
    """
    x_sequence = [token_to_idx[token] for token in seed_phrase]
    
    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        model(torch.LongTensor([ix]).cuda())
    
    # start generating
    for _ in range(max_length - len(seed_phrase)):
        probs = model(torch.LongTensor([x_sequence[-1]]).cuda())
        probs = probs.cpu()
        probs = list(np.exp(np.array(probs.data[0], dtype=np.float64)))
        probs /= sum(probs)
        x_sequence.append(np.random.choice(len(idx_to_token), p=probs))
        
    return str(''.join([idx_to_token[ix] for ix in x_sequence 
                        if idx_to_token[ix] != PAD_TOKEN and idx_to_token[ix] != SOS_TOKEN])).capitalize()

def main():
    coloredlogs.install(level='DEBUG')
    try:
        train()
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()
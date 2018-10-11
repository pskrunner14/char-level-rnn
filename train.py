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
from model import BasicRNNCell, LSTMNetwork

SOS_TOKEN = '~'
PAD_TOKEN = '#'
EOS_TOKEN = '^'

@click.command()
@click.option(
    '-f', '--filename', default='data/names',
    type=click.Path(exists=True), help='path for the training data file'
)
@click.option(
    '-rt', '--rnn-type', default='basic',
    help='type of RNN layer to use'
)
@click.option(
    '-es', '--emb-size', default=64,
    help='size of the each embedding'
)
@click.option(
    '-hs', '--hidden-size', default=256,
    help='number of hidden RNN units'
)
@click.option(
    '-n', '--num-epochs', default=50,
    help='number of epochs for training'
)
@click.option(
    '-bz', '--batch-size', default=32,
    help='number of samples per mini-batch'
)
@click.option(
    '-lr', '--learning-rate', default=0.001,
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
def train(filename, rnn_type, emb_size, hidden_size, num_epochs, batch_size, learning_rate, save_every, sample_every):
    """ Trains a character-level Recurrent Neural Network in PyTorch.

    Args: optional arguments [python train.py --help]
    """
    logging.info('reading `{}` for character sequences'.format(filename))
    inputs, token_to_idx, idx_to_token = load_dataset(file_name=filename)
    
    n_tokens = len(idx_to_token)
    max_length = inputs.size(1)
    
    logging.debug('creating char-level RNN model')
    if rnn_type == 'basic':
        model = BasicRNNCell(dropout=0.5, n_tokens=n_tokens, emb_size=emb_size, 
                            hidden_size=hidden_size, pad_id=token_to_idx[PAD_TOKEN])
    elif rnn_type == 'gru':
        pass
    elif rnn_type == 'lstm':
        model = LSTMNetwork(num_layers=2, dropout=0.5, n_tokens=n_tokens, 
                            emb_size=emb_size, hidden_size=hidden_size, 
                            pad_id=token_to_idx[PAD_TOKEN])
    else:
        raise UserWarning('unknown RNN type.')
    if torch.cuda.is_available():
        model.cuda()
    
    logging.debug('defining model training operations')
    # define training procedures and operations for training the model
    criterion = nn.NLLLoss(reduction='elementwise_mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6, 
                                                    factor=0.1, patience=7, verbose=True)

    # train-val-test split of the dataset
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
        epoch_loss, n_iter = 0.0, 0
        # loop over batches
        for batch in tqdm(iterate_minibatches(train_tensors, batchsize=batch_size),
                        desc='Epoch[{}/{}]'.format(epoch, num_epochs), leave=False,
                        total=train_tensors.size(0) // batch_size):
            # optimize model parameters
            epoch_loss += optimize(model, batch, max_length, n_tokens, criterion, optimizer)
            n_iter += 1
        # evaluate model after every epoch
        val_loss = evaluate(model, val_tensors, max_length, n_tokens, criterion)
        # lr_scheduler decreases lr when stuck at local minima 
        scheduler.step(val_loss)
        # log epoch status info
        logging.info('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'
                    .format(epoch, num_epochs, epoch_loss / n_iter, val_loss))
        
        # sample from the model every few epochs
        if epoch % sample_every == 0:
            for _ in range(10):
                sample = generate_sample(model, token_to_idx, idx_to_token, max_length, n_tokens)
                logging.debug(sample)

def optimize(model, inputs, max_length, n_tokens, criterion, optimizer):
    model.train()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, max_length, n_tokens)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    targets = inputs[:, 1: ].contiguous().view(-1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    # backpropagate error
    loss.backward()
    # update model parameters
    optimizer.step()
    return loss.item()

def evaluate(model, inputs, max_length, n_tokens, criterion):
    model.eval()
    # compute outputs after one forward pass
    outputs = forward(model, inputs, max_length, n_tokens)
    # ignore the first timestep since we don't have prev input for it
    # (timesteps, batches, 1) -> (timesteps x batches x 1)
    targets = inputs[:, 1: ].contiguous().view(-1)
    # compute loss wrt targets
    loss = criterion(outputs, targets)
    return loss.item()

def forward(model, inputs, max_length, n_tokens):
    if torch.cuda.is_available():
        inputs.cuda()
    # tensor for storing outputs of each time-step
    outputs = torch.Tensor(max_length, inputs.size(0), n_tokens)
    # loop over time-steps
    for t in range(max_length):
        # t-th time-step input
        input_t = inputs[:, t]
        outputs[t] = model(input_t.cuda())
    # (timesteps, batches, n_tokens) -> (batches, timesteps, n_tokens)
    outputs = outputs.permute(1, 0, 2)
    # ignore the last time-step since we don't have a target for it.
    outputs = outputs[:, :-1, :]
    # (batches, timesteps, n_tokens) -> (batches x timesteps, n_tokens)
    outputs = outputs.contiguous().view(-1, n_tokens)
    return outputs

def generate_sample(model, token_to_idx, idx_to_token, max_length, n_tokens, seed_phrase=SOS_TOKEN):
    """ Generates samples using seed phrase.

    Args:
        model (nn.Module): the character-level RNN model to use for sampling.
        token_to_idx (dict of `str`: `int`): character to token_id mapping dictionary (vocab).
        idx_to_token (list of `str`): index (token_id) to character mapping list (vocab).
        max_length (int): max length of a sequence to sample using model.
        seed_phrase (str): the initial seed characters to feed the model. If unspecified, defaults to `SOS_TOKEN`.
    
    Returns:
        str: generated sample from the model using the seed_phrase.
    """
    model.eval()
    if seed_phrase[0] != SOS_TOKEN:
        seed_phrase = SOS_TOKEN + seed_phrase
    # convert to token ids for model
    sequence = [token_to_idx[token] for token in seed_phrase]
    input_tensor = torch.LongTensor(sequence)
    if torch.cuda.is_available():
        input_tensor.cuda()
    # feed the seed phrase to manipulate rnn hidden states
    for i in range(len(sequence) - 1):
        model(input_tensor[i])
    
    # start generating
    for _ in range(max_length - len(seed_phrase)):
        # sample char from previous time-step
        input_tensor = torch.LongTensor([sequence[-1]])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        probs = model(input_tensor)
        # need to use `exp` as output is `LogSoftmax`
        probs = list(np.exp(np.array(probs.data[0])))
        # normalize probabilities to ensure sum = 1
        probs /= sum(probs)
        # sample char randomly based on probabilities
        sequence.append(np.random.choice(len(idx_to_token), p=probs))
    return str(''.join([idx_to_token[ix] for ix in sequence 
                    if idx_to_token[ix] != PAD_TOKEN and idx_to_token[ix] != SOS_TOKEN])).capitalize()

def main():
    coloredlogs.install(level='DEBUG')
    try:
        train()
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()
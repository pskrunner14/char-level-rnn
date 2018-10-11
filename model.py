import torch
import torch.nn as nn

def glorot_normal_initializer(m):
    """ Applies Glorot Normal initialization to layer parameters.
    
    "Understanding the difficulty of training deep feedforward neural networks" 
    by Glorot, X. & Bengio, Y. (2010)

    Args:
        m (nn.Module): a particular layer whose params are to be initialized.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class BasicRNNCell(nn.Module):
    """ Basic recurrent neural network cell.

    This module imitates the basic structure and flow of an 
    RNN cell with embeddings, hidden states and output softmax.

    Args:
        n_tokens (int): number of unique tokens in corpus.
        emb_size (int): dimensionality of each embedding.
        hidden_size (int): number of hidden units in RNN hidden layer.
        pad_id (int): token_id of the padding token.

    Attributes:
        net (nn.Sequential): RNN sequential model.
    """

    def __init__(self, dropout, n_tokens, emb_size, hidden_size, pad_id):
        super(BasicRNNCell, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_tokens, emb_size, padding_idx=pad_id),
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_tokens),
            nn.LogSoftmax(dim=1)
        )
        self.net.apply(glorot_normal_initializer)

    def forward(self, inputs):
        """ Implements the forward pass of the model.

        Args:
            inputs (torch.LongTensor): input token sequences to feed the network.
        Returns:
            torch.Tensor: output log softmax probability distribution over tokens.
        """
        return self.net.forward(inputs)
        
class LSTMNetwork(nn.Module):
    """ LSTM recurrent neural network.

    This module imitates the complex structure and 
    flow of a Long-Short Term Memory RNN network.

    Args:
        num_layers (int): number of LSTM layers in RNN model.
        dropout (float): probability of shutting down a neuron in RNN layers.
        n_tokens (int): number of unique tokens in corpus.
        emb_size (int): dimensionality of each embedding.
        hidden_size (int): number of hidden units in RNN hidden layer.
        pad_id (int): token_id of the padding token.

    Attributes:
        
    """

    def __init__(self, num_layers, dropout, n_tokens, emb_size, hidden_size, pad_id):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(n_tokens, emb_size, padding_idx=pad_id)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, 
                        num_layers=num_layers, dropout=dropout, batch_first=True)
        self.act = nn.LeakyReLU(0.2)
        self.logits = nn.Linear(hidden_size, n_tokens)
        self.out = nn.LogSoftmax(dim=1)
        
        self.logits.apply(glorot_normal_initializer)

    def forward(self, inputs):
        pass

    def initHidden(self):
        pass
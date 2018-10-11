import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class CharRNN(nn.Module):

    def __init__(self, n_tokens, emb_size, hidden_size, pad_id):
        super(CharRNN, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(n_tokens, emb_size, padding_idx=pad_id),
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_tokens),
            nn.LogSoftmax(dim=1)
        )
        self.net.apply(init_weights)

    def forward(self, inputs):
        return self.net.forward(inputs)
        
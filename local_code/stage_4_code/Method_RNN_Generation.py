"""Embedding -> causal recurrent unit -> vocabulary logits at every time step."""
from torch import nn
from local_code.base_class.method import method


class Method_RNN_Generation(method, nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 num_layers=1, rnn_type='rnn', dropout=0.0):
        method.__init__(self, rnn_type + ' generator', '')
        nn.Module.__init__(self)
        self.config = dict(vocab_size=vocab_size, embedding_dim=embedding_dim,
                           hidden_dim=hidden_dim, num_layers=num_layers,
                           rnn_type=rnn_type, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        unit = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.recurrent = unit(embedding_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, hidden=None):
        states, hidden = self.recurrent(self.embedding(ids), hidden)
        return self.output(self.dropout(states)), hidden

    def run(self, ids):
        return self.forward(ids)

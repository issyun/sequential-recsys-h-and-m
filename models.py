import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU4Rec(nn.Module):
    def __init__(self, num_articles, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_articles, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden_size, num_articles)
        self.init_weight()
        
    def forward(self, input, lengths):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.proj(output)
        return output
    
    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.proj.weight.data.uniform_(-0.1, 0.1)
        self.proj.bias.data.fill_(0)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU4Rec(nn.Module):
    def __init__(self, num_items, num_users, item_emb_size, user_emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(num_items, item_emb_size)
        self.user_embedding = nn.Embedding(num_users, user_emb_size)
        self.gru = nn.GRU(item_emb_size + user_emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden_size, num_items)
        self.init_weight()
        
    def forward(self, input, lengths, users):
        item_emb = self.item_embedding(input)
        user_emb = self.user_embedding(users).unsqueeze(1).repeat(1, item_emb.size(1), 1)
        emb = torch.cat([item_emb, user_emb], dim=-1)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.proj(output)
        return output
    
    def init_weight(self):
        self.item_embedding.weight.data.uniform_(-0.1, 0.1)
        self.user_embedding.weight.data.uniform_(-0.1, 0.1)
        self.proj.weight.data.uniform_(-0.1, 0.1)
        self.proj.bias.data.fill_(0)
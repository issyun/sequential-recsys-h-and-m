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

class GRU4RecNoUser(nn.Module):
    def __init__(self, num_items, item_emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_items, item_emb_size)
        self.gru = nn.GRU(item_emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden_size, num_items)
        self.init_weight()
        
    def forward(self, input, lengths):
        emb = self.embedding(input)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.proj(output)
        return output
    
    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.proj.weight.data.uniform_(-0.1, 0.1)
        self.proj.bias.data.fill_(0)

class NARM(nn.Module):
    def __init__(self, num_items, emb_size, hidden_size, num_layers=1):
        super(NARM, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_size = emb_size

        self.emb = nn.Embedding(num_items, emb_size)
        self.emb_do = nn.Dropout(0.25)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True)
        self.A_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.A_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.ct_do = nn.Dropout(0.5)
        self.B = nn.Linear(2 * hidden_size, emb_size, bias=False)

    def forward(self, input, lengths):
        emb = self.emb_do(self.emb(input))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        ht = hidden[-1]
        c_global = ht
        q1 = self.A_1(gru_out)
        q2 = self.A_2(ht)
        q2 = q2.unsqueeze(1).expand_as(q1)
        alpha = self.v(torch.sigmoid(q1 + q2).reshape(-1, self.hidden_size)).reshape(input.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = self.B(torch.cat([c_local, c_global], 1)).unsqueeze(2)
        c_t = self.ct_do(c_t)
        item_embs = self.emb(torch.arange(self.num_items).to(input.device)).unsqueeze(0).expand(input.shape[0], -1, -1)
        scores = torch.bmm(item_embs, c_t).squeeze()

        return scores
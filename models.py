import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, num_articles, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_articles, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.proj_1 = nn.Linear(hidden_size, num_articles//2)
        self.proj_2 = nn.Linear(num_articles//2, num_articles)
        self.init_weight()
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, _ = self.gru(embedded, hidden)
        output = self.proj_1(output).relu()
        output = self.proj_2(output)
        return output
    
    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.proj_1.weight.data.uniform_(-0.1, 0.1)
        self.proj_1.bias.data.fill_(0)
        self.proj_2.weight.data.uniform_(-0.1, 0.1)
        self.proj_2.bias.data.fill_(0)
import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.proj_1 = nn.Linear(hidden_size, output_size//2)
        self.proj_2 = nn.Linear(output_size//2, output_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
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
import torch
import torch.nn as nn
import random

class Model(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, vocab_size=259, num_layers=2, droput_rate=.2):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(droput_rate)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = 1

    def init_hidden_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    
    def forward(self, song, mask, hidden=None, teacher_forcing_ratio=None):
        if hidden is None:
            hidden = self.init_hidden_state()
            
        output = []
        x = song[0]
        
        for i in range(len(song)):
            x = self.embedding(torch.tensor(x, dtype=torch.long))
            x = x.unsqueeze(0).unsqueeze(0)
            curr, hidden = self.lstm(x, hidden)
            curr = self.dropout(curr)
            curr = self.fc(curr)
            
            if mask[i]:
                clamped_output = torch.zeros_like(curr)
                clamped_output[0, 0, song[i]] = 1.0
                output.append(clamped_output)
                curr = song[i]
            else: 
                if teacher_forcing_ratio and random.random() < teacher_forcing_ratio:
                    output.append(curr)
                    curr = song[i]
                else:
                    output.append(curr)
                    curr = curr.argmax(dim=2).item()
            x = curr
        output = torch.cat(output, dim=1).view(-1, self.fc.out_features)
        return output, hidden
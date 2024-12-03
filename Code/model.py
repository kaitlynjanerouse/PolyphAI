import torch
import torch.nn as nn
import random

class Model(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=512, vocab_size=97, num_layers=2, dropout_rate=0.15, device="cpu"):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = 1
        self.device = device

    def init_hidden_state(self):
        # Move the hidden states to the device
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))
    
    def forward(self, song, mask, hidden=None, teacher_forcing_ratio=None):
        if hidden is None:
            hidden = self.init_hidden_state()

        output = []
        x = song[0]

        start_logits = torch.zeros(1, 1, self.fc.out_features, device=self.device)
        start_logits[0, 0, song[0]] = 1.0
        output.append(start_logits)

        for i in range(1, len(song)):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.long, device=self.device)
            else:
                x = x.to(dtype=torch.long, device=self.device)
            x = self.embedding(x)

            x = x.unsqueeze(0).unsqueeze(0)
            curr, hidden = self.lstm(x, hidden)
            curr = self.dropout(curr)
            curr = self.fc(curr)
            
            if mask[i]:
                clamped_output = torch.zeros_like(curr).to(self.device)
                clamped_output[0, 0, song[i]] = 1.0
                output.append(clamped_output)
                curr = song[i]
                x = torch.tensor(song[i], dtype=torch.long, device=self.device)  # Use clamped value as next input
                x_embedded = self.embedding(x).unsqueeze(0).unsqueeze(0)
                _, hidden = self.lstm(x_embedded, hidden)

            else: 
                if teacher_forcing_ratio and random.random() < teacher_forcing_ratio:
                    output.append(curr)
                    curr = song[i]
                    x = torch.tensor(song[i], dtype=torch.long, device=self.device)  # Use clamped value as next input
                    x_embedded = self.embedding(x).unsqueeze(0).unsqueeze(0)
                    _, hidden = self.lstm(x_embedded, hidden)
                else:
                    output.append(curr)
                    curr = curr.argmax(dim=2).item()
            x = curr
        output = torch.cat(output, dim=1).view(-1, self.fc.out_features)
        return output, hidden
    
    
    
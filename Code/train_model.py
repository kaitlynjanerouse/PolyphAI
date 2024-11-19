import torch.nn as nn
import torch

class TrainModel():
    def __init__(self, model, train_set, validation_set, notes_in_data, device="cpu"):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.num_epochs = 100
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.notes_in_data = notes_in_data
        self.device = device 

    def train_model(self):
        token_dictionary = self.embedding_dictionary()
        for index, song in enumerate(self.train_set):
            self.model.train()
            melody_mask = torch.tensor(self.compute_mask(song), device=self.device)
            embeded_song = torch.tensor([token_dictionary[token] for token in song], device=self.device)
            hidden = None
            for epoch in range(self.num_epochs):
                teacher_forcing_rate = max(0.5 * (1 - epoch / self.num_epochs), 0)
                self.optimizer.zero_grad()
                output, hidden = self.model(embeded_song, melody_mask, hidden, teacher_forcing_rate)
                loss = self.criterion(output, embeded_song)
                loss.backward()
                hidden = tuple(h.detach().to(self.device) for h in hidden)  
                self.optimizer.step()
                if (epoch + 1) % 1 == 0:
                    print(f"Song {index + 1}, Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}")
                    
            with torch.no_grad():
                self.model.eval()
                total_val_loss = 0
                for val_song in self.validation_set:
                    val_embeded_song = torch.tensor(
                        [token_dictionary[token] for token in val_song], device=self.device
                    )
                    val_melody_mask = torch.tensor(self.compute_mask(val_song), device=self.device)
                    val_input_song = torch.tensor(
                        self.harmonies_to_zero(val_embeded_song), device=self.device
                    )
                    val_output, _ = self.model(val_input_song, val_melody_mask)
                    val_loss = self.criterion(val_output, val_embeded_song)
                    total_val_loss += val_loss.item()
                print(f"Validation loss: {total_val_loss / len(self.validation_set)}")

    def compute_mask(self, song):
        result = [True]  # start
        for i in range(1, len(song) - 1):
            if i % 5 == 1 or i % 5 == 0:
                result.append(True)
            else:
                result.append(False)
        result.append(True)  # end
        return result

    def embedding_dictionary(self):
        token_to_index = {("START"): 0, ("END"): 1, ("|||"): 2, (0, 0): 3, (0, 1): 4}
        non_zero_notes = {x for x in self.notes_in_data if x != 0}
        min_val = min(non_zero_notes)
        max_val = max(self.notes_in_data)
        range_offset = max_val - min_val + 1
        base_index = 5
        for note in range(min_val, max_val + 1):
            token_to_index[(note, 0)] = base_index + note - min_val
            token_to_index[(note, 1)] = base_index + note - min_val + range_offset
        return token_to_index

    def harmonies_to_zero(self, song):
        result = []
        for i in range(len(song)):
            if i % 5 in (2, 3, 4):
                result.append(-1)
            else:
                result.append(song[i])
        return result
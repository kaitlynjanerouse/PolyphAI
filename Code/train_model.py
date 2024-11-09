import torch.nn as nn
import torch

class TrainModel():
    def __init__(self, model, train_set, validation_set):
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 120
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
    
    def train_model(self):
        token_dictionary = self.embedding_dictionary()
        for index, song in enumerate(self.train_set[:10]):
            self.model.train()
            melody_mask = self.compute_mask_training(song)
            embeded_song = [token_dictionary[token] for token in song]
            hidden = None
            for epoch in range(self.num_epochs):
                teacher_forcing_rate = max(0.5 * (1 - epoch / self.num_epochs), 0)
                self.optimizer.zero_grad()
                output, hidden = self.model(embeded_song, melody_mask, hidden, teacher_forcing_rate)
                loss = self.criterion(output, torch.tensor(embeded_song))
                loss.backward()
                hidden = tuple(h.detach() for h in hidden)
                self.optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Song {index + 1}, Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}")
                    
            with torch.no_grad():
                self.model.eval()
                total_val_loss = 0
                for val_song in self.validation_set:
                    val_embeded_song = [token_dictionary[token] for token in val_song]
                    val_melody_mask = self.compute_mask_testing(val_song)
                    val_input_song = self.harmonies_to_zero(val_embeded_song)
                    val_output, _ = self.model(val_input_song, val_melody_mask)
                    val_loss = self.num_epochscriterion(val_output, torch.tensor(val_embeded_song))
                    total_val_loss += val_loss.item()
                print(f'Validation loss: {total_val_loss / len(self.validation_set)}')

    def compute_mask_training(self, song):
        result = []
        result.append(True) # start
        for i in range(1, len(song) - 1):
            if i % 5 == 1 or i % 5 == 0:
                result.append(True)
            else:
                result.append(False)
        result.append(True) # end
        return result

    def compute_mask_testing(self, song):
        result = []
        for i in range(len(song)):
            if i % 5 == 1 and i != len(song) - 1:
                result.append(True)
            else:
                result.append(False)
        return result

    def embedding_dictionary(self):
        token_to_index = {("START"): 256, ("END"): 257, ("|||"): 258}
        for note in range(128):
            token_to_index[(note, 0)] = note
            token_to_index[(note, 1)] = note + 128
        return token_to_index

    def harmonies_to_zero(self, song):
        result = []
        for i in range(len(song)):
            if i % 5 == 2 or i % 5 == 3 or i % 5 == 4:
                result.append(-1)
            else:
                result.append(song[i])
        return result
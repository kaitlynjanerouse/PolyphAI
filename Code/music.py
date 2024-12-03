from music21 import stream, note
import torch

class Music():
    def __init__(self, test_set, notes):
        self.songs = [song for song in test_set]
        self.notes_in_data = notes
        self.token_dictionary = self.embedding_dictionary()

    def kickoff_model(self, model):
        for i, song in enumerate(self.songs):
            embedded_test = [self.token_dictionary[token] for token in song]
            test_mask = self.compute_mask_testing(song)
            input_embedded_test = self.harmonies_to_zero(embedded_test)

            output, _ = model(input_embedded_test, test_mask)
            self.output_to_sheet_music(output, f'output{i}.xml')
    
    def compute_mask_testing(self, song):
        result = [True]  # start
        for i in range(1, len(song) - 1):
            if i % 5 == 1 or i % 5 == 0:
                result.append(True)
            else:
                result.append(False)
        result.append(True)  # end
        return result

    def harmonies_to_zero(self, song):
        result = []
        for i in range(len(song)):
            if i % 5 == 2 or i % 5 == 3 or i % 5 == 4:
                result.append(-1)
            else:
                result.append(song[i])
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

    def midi_to_note(self, part):
        result = stream.Part()
        count = 1
        prev = round(part[0])
        
        for i in range(1, len(part)):
            curr = round(part[i])
            if curr == prev:
                count += 1
            else:
                result.append(note.Note(prev, quarterLength=count / 4))
                count = 1
            prev = curr
        result.append(note.Note(prev, quarterLength=count / 4))
        return result

    def process_sequence(self, sequence, delimiter_token="|||"):
        index_to_token = {v: k for k, v in self.token_dictionary.items()}
        original_sequence = [index_to_token[embedded_value] for embedded_value in sequence]
        original_sequence = original_sequence[1:-1]
        melody, alto, tenor, bass = [], [], [], []
        for i in range(0, len(original_sequence), 5):
            melody.append(original_sequence[i][0])
            alto.append(original_sequence[i+1][0])
            tenor.append(original_sequence[i+2][0])
            bass.append(original_sequence[i+3][0])
        
        return melody, alto, tenor, bass

    def output_to_sheet_music(self, result, file_name):
        result = torch.argmax(result, dim=-1) 
        result = result.detach().cpu().numpy()

        melody_notes, alto_notes, tenor_notes, bass_notes = self.process_sequence(result)

        melody_part = self.midi_to_note(melody_notes)
        alto_part = self.midi_to_note(alto_notes)
        tenor_part = self.midi_to_note(tenor_notes)
        bass_part = self.midi_to_note(bass_notes)

        score = stream.Score()
        score.append(melody_part)
        score.append(alto_part)
        score.append(tenor_part)
        score.append(bass_part)

        # score.show('midi')
        score.write('musicxml', f'Results/{file_name}')
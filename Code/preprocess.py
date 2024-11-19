import os
import pandas as pd 
import numpy as np

class Preprocess():
    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.validation_set = []
        self.notes = set()

    def process_data(self):
        folder_path = 'Data/'
        for dirname in os.listdir(folder_path):
            if dirname != '.DS_Store':
                for filename in os.listdir(folder_path + dirname):
                    if filename != '.ipynb_checkpoints':
                        df = pd.read_csv(folder_path + dirname + '/' + filename)
                        transpose = self.key_transposition(df)
                        song = self.encode_song(transpose)
                        if dirname == 'test':
                            self.test_set.append(song)
                        if dirname == 'train':
                            self.train_set.append(song)
                        if dirname == 'valid':
                            self.validation_set.append(song)

    def get_pitch_class(self, note):
        return note % 12

    def find_matching_octave_note(self, df):
        bass_line = df['note3'].values
        if bass_line[0] == 48 or bass_line[0] == 45:
            return bass_line[0]
        last_bass_note = bass_line[-1]
        return last_bass_note

    def explore_for_lowest_tonic(self, df, pitch_class):
        bass_notes = df['note3'].values  
        matching_notes = [note for note in bass_notes if self.get_pitch_class(note) == pitch_class]
        if 0 in matching_notes:
            matching_notes = [i for i in matching_notes if i != 0]
        if matching_notes:
            return min(matching_notes) 
        else:
            return bass_notes[0]

    def detect_tonic(self, df):
        candidate_note = self.find_matching_octave_note(df)
        pitch_class = self.get_pitch_class(candidate_note)
        true_tonic_note = self.explore_for_lowest_tonic(df, pitch_class)
        return candidate_note if (candidate_note == 45 or candidate_note == 48) else true_tonic_note

    def key_transposition(self, df):
        return self.normalize_to_standard_range(df)
    
    def normalize_to_standard_range(self, df):
        tonic_note = self.detect_tonic(df)
        target_pitch_class = 0 
        transpose_val = target_pitch_class - self.get_pitch_class(tonic_note)

        desired_min = 36
        desired_max = 84 
        
        def transpose_and_wrap(note):
            if note == 0: 
                return 0
            transposed = note + transpose_val

            while transposed < desired_min:
                transposed += 12
            while transposed > desired_max:
                transposed -= 12
            return transposed

        df = df.applymap(transpose_and_wrap)
        return df


    def encode_song(self, song):
        
        result = []
        prev = {'note0': -1, 'note1': -1, 'note2': -1, 'note3': -1}
        result.append('START')

        for index, row in song.iterrows():
            for voice in ['note0', 'note1', 'note2', 'note3']:
                pitch = row[voice]
                previous_pitch = prev[voice]
                
                tied = 1 if pitch == previous_pitch else 0
                result.append((pitch,tied))
                prev[voice] = pitch
                self.notes.add(pitch)
            result.append('|||')
        result.append('END')
        return result
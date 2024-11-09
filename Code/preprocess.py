import os
import pandas as pd 

class Preprocess():
    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.validation_set = []

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
        last_bass_note = bass_line[-1]
        return last_bass_note

    def explore_for_lowest_tonic(self, df, pitch_class):
        bass_notes = df['note3'].values  
        matching_notes = [note for note in bass_notes if self.get_pitch_class(note) == pitch_class]
        if matching_notes:
            return min(matching_notes) 
        else:
            return bass_notes[0]

    def detect_tonic(self, df):
        candidate_note = self.find_matching_octave_note(df)
        pitch_class = self.get_pitch_class(candidate_note)
        true_tonic_note = self.explore_for_lowest_tonic(df, pitch_class)
        return true_tonic_note

    def is_major(self, df):
        chord_notes = df.iloc[0, :4].values
        unique_pitch_classes = set(note % 12 for note in chord_notes)
        
        for root in unique_pitch_classes:
            intervals = sorted((note - root) % 12 for note in unique_pitch_classes if note != root)
            if 4 in intervals and 7 in intervals:
                return True
            
            if 3 in intervals and 7 in intervals:
                return False
        
        return False

    def key_transposition(self, df):
        tonic_note = self.detect_tonic(df)
        transpose_val = 48 if self.is_major(df) else 45
        transpose_val -= tonic_note 
        df = (df + transpose_val).clip(lower=0, upper=127)
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
            result.append('|||')
        result.append('END')
        return result
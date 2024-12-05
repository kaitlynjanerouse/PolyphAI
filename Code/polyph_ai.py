"""
Driver class to kickoff the model. This class either trains the model or 
loads in a trained model to process user input.
"""

import os
import argparse
import torch
from preprocess import Preprocess
from model import Model
from train_model import TrainModel
from music import Music

class PolyphAI:
    def __init__(self, csv_path=None, retrain=False):
        self.csv_path = csv_path
        self.retrain = retrain
        self.model_path = "model_checkpoint.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None

    def setup_device(self):
        if torch.cuda.is_available():
            print("GPU is available!")
        else:
            print("GPU is not available, using CPU")

    def load_or_train_model(self):
        self.preprocess = Preprocess(self.csv_path)
        self.preprocess.process_data()

        self.model = Model(device=self.device).to(self.device)

        if self.retrain or not os.path.exists(self.model_path):
            if self.retrain:
                print("Forced retraining...")
            else:
                print("No saved model found. Training a new model...")
            train = TrainModel(self.model, self.preprocess.train_set, self.preprocess.validation_set, self.preprocess.notes, self.device)
            train.train_model()
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Finished model stored in {self.model_path}")
        else:
            print("Loading saved model...")
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model.eval()

    def kickoff_music_generation(self):
        path = ''
        if self.csv_path:
            path = 'User Results'
            music = Music(self.preprocess.user_input, self.preprocess.notes, 'User Results')
        else:
            path = 'Results'
            music = Music(self.preprocess.test_set, self.preprocess.notes, 'Results')
        music.kickoff_model(self.model)
        print(f'Results stored in {path} folder')

    def run(self):
        self.setup_device()
        self.load_or_train_model()
        self.kickoff_music_generation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to the .csv file containing the melody notes. If none is provided, the model will generate results for the test set."
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of the model even if a saved model exists."
    )

    args = parser.parse_args()

    polyph_ai = PolyphAI(args.csv_path, args.retrain)
    polyph_ai.run()
from preprocess import Preprocess
from model import Model
from train_model import TrainModel
from music import Music
import torch

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

def polyph_ai():
    preprocess = Preprocess()
    preprocess.process_data()
    model = Model()
    train = TrainModel(model, preprocess.train_set, preprocess.validation_set, preprocess.notes)
    train.train_model()
    music = Music(preprocess.test_set, preprocess.notes)
    music.kickoff_model(model)


if __name__ == "__main__":
    polyph_ai()
from preprocess import Preprocess
from model import Model
from train_model import TrainModel
from music import Music
import torch

def polyph_ai():
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess()
    preprocess.process_data()
    model = Model(device=device).to(device)
    train = TrainModel(model, preprocess.train_set, preprocess.validation_set, preprocess.notes, device)
    train.train_model()
    train.plot_losses()
    music = Music(preprocess.test_set, preprocess.notes)
    music.kickoff_model(model)


if __name__ == "__main__":
    polyph_ai()
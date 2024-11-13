from preprocess import Preprocess
from model import Model
from train_model import TrainModel
from music import Music

# comment
def polyph_ai():
    preprocess = Preprocess()
    preprocess.process_data()
    model = Model()
    train = TrainModel(model, preprocess.train_set, preprocess.validation_set)
    train.train_model()
    music = Music(preprocess.test_set[0])
    music.kickoff_model(model)


if __name__ == "__main__":
    polyph_ai()
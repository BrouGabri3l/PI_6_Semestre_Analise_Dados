import os
import pickle
from recommender import Recommender

def load_recommender(path='models'):
    return Recommender(model_dir=path)
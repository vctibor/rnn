
import gzip
import pickle

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)



data = load_data()
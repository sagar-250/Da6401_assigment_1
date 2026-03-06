import numpy as np
from tensorflow import keras    


def load_data(dataset='mnist', normalize=True, flatten=True):
    if dataset.lower() == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset.lower() == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    if normalize:
        X_train=X_train.astype('float32')/255.0
        X_test=X_test.astype('float32')/255.0
    
    if flatten:
        X_train=X_train.reshape(X_train.shape[0], -1)
        X_test=X_test.reshape(X_test.shape[0], -1)
    
    return (X_train, y_train), (X_test, y_test)

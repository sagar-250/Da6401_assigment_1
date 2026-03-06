import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD
from utils.data_loader import load_data

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

def main():
    # Load data
    (X_train, y_train), (X_test, y_test) = load_data(dataset='mnist')
    
    # Use only a small subset for quick testing
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_test_small = X_test[:200]
    y_test_small = y_test[:200]
    
    # One-hot encode labels
    y_train_oh = one_hot(y_train_small)
    y_test_oh = one_hot(y_test_small)
    
    # Create a very small network: 784 -> 32 -> 10
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[32],  # Just 1 hidden layer with 32 neurons
        output_size=10,
        activation='relu',
        weight_init='xavier',
        loss='cross_entropy'
    )
    
    # Simple SGD optimizer
    optimizer = SGD(lr=0.01)
    
    print("Training for 1 epoch...")
    model.train(
        X_train_small, 
        y_train_oh, 
        optimizer,
        epochs=1,  # Just 1 pass
        batch_size=32,
        X_val=X_test_small, 
        y_val=y_test_oh
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_small, y_test_oh)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
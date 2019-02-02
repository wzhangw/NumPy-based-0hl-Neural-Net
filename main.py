import pandas as pd
import numpy as np
from numpy_based_0hl_neural_network import NumPyBased0hlNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def read_from_file(filename):
    rows = []
    with open(filename, "r") as file:
        for line in file:
            row = []
            for letter in line.strip():
                digit = int(letter)
                row.append(digit)
            row = np.array(row)
            rows.append(row)
    data = np.array(rows)
    X_origin = data[:, 0: -1]
    Y_origin = data[:, -1: data.shape[1]]
    return data, X_origin, Y_origin

def convert_labels_to_onehot(Y_origin):
    min_class = np.min(Y_origin)
    max_class = np.max(Y_origin)
    Y_onehot = np.zeros((Y_origin.shape[0], max_class - min_class + 1))
    for row in range(Y_origin.shape[0]):
        label = Y_origin[row, 0]
        Y_onehot[row, label] = 1
    return Y_onehot

def convert_onehot_to_labels(Y_onehot):
    Y_origin = np.argmax(Y_onehot, axis=0)
    return Y_origin

def main(debug_mode=True, cost_plot_mode=True):
    data, X_origin, Y_origin = read_from_file("good-moves.txt")
    if debug_mode:
        print("data.shape = " + str(data.shape))
        print("X_origin.shape = " + str(X_origin.shape))
        print("Y_origin.shape = " + str(Y_origin.shape))
    Y_onehot = convert_labels_to_onehot(Y_origin)
    if debug_mode:
        print("Y_onehot.shape = " + str(Y_onehot.shape))
    X_train, X_test, Y_train, Y_test = train_test_split(X_origin, Y_onehot, test_size=0.2, random_state=0)
    X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T
    if debug_mode:
        print("X_train.shape:\t " + str(X_train.shape))
        print("X_test.shape:\t " + str(X_test.shape))
        print("Y_train.shape:\t " + str(Y_train.shape))
        print("Y_test.shape:\t " + str(Y_test.shape))
    neural_network = NumPyBased0hlNeuralNetwork()
    neural_network.fit(X=X_train, Y=Y_train, decay_rate=0.0, early_stopping_point=10000, convergence_tolerance = 0.0001, batch_size=Y_train.shape[1], debug_mode=debug_mode, cost_plot_mode=cost_plot_mode)
    predicted_classes, predicted_onehots = neural_network.predict(X=X_test, debug_mode=debug_mode)
    F1_score = f1_score(convert_onehot_to_labels(Y_test), predicted_classes, average="weighted")
    print("Test set F1 score = " + str(F1_score))

main()

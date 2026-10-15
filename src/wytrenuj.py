import tkinter as tk
import numpy as np
import kagglehub
import idx2numpy
import os

###
learning_rate = 0.005
epochs = 20
hidden_size = 100
pixels_per_image = 784
num_labels = 10
###

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

x_train = idx2numpy.convert_from_file(os.path.join(path, "train-images.idx3-ubyte"))
x_train = x_train.reshape(-1, 28 * 28)
y_train = idx2numpy.convert_from_file(os.path.join(path, "train-labels.idx1-ubyte"))

x_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-images.idx3-ubyte"))
x_test = x_test.reshape(-1, 28 * 28)
y_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-labels.idx1-ubyte"))

training_sample, test_sample = 1000, 1000
training_images = x_train[0:training_sample] / 255
test_images = x_test[0:test_sample] / 255

def one_hot_encoding(labels, dimension=10):
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    return one_hot_labels.astype(np.float64)

training_labels = one_hot_encoding(y_train[:training_sample])
test_labels = one_hot_encoding(y_test[:test_sample])

seed = 884736743
rng = np.random.default_rng(seed)

def relu(x):
    return (x >= 0) * x

def relu2deriv(output):
    return output >= 0


weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1
weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1

store_training_loss = []
store_training_accurate_pred = []
store_test_loss = []
store_test_accurate_pred = []

for j in range(epochs):
    training_loss = 0.0
    training_accurate_predictions = 0

    for i in range(len(training_images)):
        layer_0 = training_images[i]

        layer_1 = np.dot(layer_0, weights_1)

        layer_1 = relu(layer_1)

        dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)

        layer_1 *= dropout_mask * 2

        layer_2 = np.dot(layer_1, weights_2)

        training_loss += np.sum((training_labels[i] - layer_2) ** 2)

        training_accurate_predictions += int(
            np.argmax(layer_2) == np.argmax(training_labels[i])
        )

        layer_2_delta = training_labels[i] - layer_2

        layer_1_delta = np.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)

        layer_1_delta *= dropout_mask

        weights_1 += learning_rate * np.outer(layer_0, layer_1_delta)
        weights_2 += learning_rate * np.outer(layer_1, layer_2_delta)

    store_training_loss.append(training_loss)
    store_training_accurate_pred.append(training_accurate_predictions)

    results = relu(test_images @ weights_1) @ weights_2

    test_loss = np.sum((test_labels - results) ** 2)

    test_accurate_predictions = np.sum(
        np.argmax(results, axis=1) == np.argmax(test_labels, axis=1)
    )

    store_test_loss.append(test_loss)
    store_test_accurate_pred.append(test_accurate_predictions)

    print(
        (
            f"Epoch: {j}\n"
            f"  Training set error: {training_loss / len(training_images):.3f}\n"
            f"  Training set accuracy: {training_accurate_predictions / len(training_images)}\n"
            f"  Test set error: {test_loss / len(test_images):.3f}\n"
            f"  Test set accuracy: {test_accurate_predictions / len(test_images)}"
        )
    )

np.savez("model.npz", weights_1=weights_1, weights_2=weights_2)
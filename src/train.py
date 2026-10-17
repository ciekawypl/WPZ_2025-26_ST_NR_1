import os
import numpy as np
import kagglehub
import idx2numpy

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

x_train = idx2numpy.convert_from_file(os.path.join(path, "train-images.idx3-ubyte"))
x_train = x_train.reshape(-1, 28 * 28)
y_train = idx2numpy.convert_from_file(os.path.join(path, "train-labels.idx1-ubyte"))

x_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-images.idx3-ubyte"))
x_test = x_test.reshape(-1, 28 * 28)
y_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-labels.idx1-ubyte"))

training_images = x_train / 255
test_images = x_test / 255

def one_hot_encoding(labels, dimension=10):
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    return one_hot_labels.astype(np.float64)

training_labels = one_hot_encoding(y_train)
test_labels = one_hot_encoding(y_test)


np.random.seed(42)
input_size = 784
hidden_size = 512
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

learning_rate = 0.05
epochs = 50
batch_size = 64

for epoch in range(epochs):
    permutation = np.random.permutation(training_images.shape[0])
    training_images_shuffled = training_images[permutation]
    y_train_shuffled = training_labels[permutation]
    
    for i in range(0, training_images.shape[0], batch_size):
        X_batch = training_images_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        preds = softmax(z2)

        error = preds - y_batch
        dW2 = np.dot(a1.T, error) / batch_size
        db2 = np.sum(error, axis=0, keepdims=True) / batch_size
        
        d_hidden = np.dot(error, W2.T) * (z1 > 0)
        dW1 = np.dot(X_batch.T, d_hidden) / batch_size
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / batch_size

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    z1_test = np.dot(test_images, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    test_preds = np.argmax(softmax(z2_test), axis=1)
    accuracy = np.mean(test_preds == y_test) * 100
    
    print(f"Epoka {epoch + 1}/{epochs} | Dokładność testowa: {accuracy:.2f}%")

np.savez('model.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Wagi zapisane w 'model.npz'.")
import os
import gzip
import requests
import numpy as np

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

data_dir = "./mnist_data"
os.makedirs(data_dir, exist_ok=True)
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print(f"Pobieranie pliku: {fname}...")
        resp = requests.get(base_url + fname)
        with open(fpath, "wb") as f:
            f.write(resp.content)

mnist_dataset = {}
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as f:
        mnist_dataset[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28) / 255.0

for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as f:
        mnist_dataset[key] = np.frombuffer(f.read(), np.uint8, offset=8)

X_train = mnist_dataset["training_images"]
y_train_raw = mnist_dataset["training_labels"]

X_test = mnist_dataset["test_images"]
y_test_raw = mnist_dataset["test_labels"]

def one_hot(labels):
    res = np.zeros((labels.size, 10))
    res[np.arange(labels.size), labels] = 1
    return res

y_train = one_hot(y_train_raw)

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
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
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

    z1_test = np.dot(X_test, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    test_preds = np.argmax(softmax(z2_test), axis=1)
    accuracy = np.mean(test_preds == y_test_raw) * 100
    
    print(f"Epoka {epoch + 1}/{epochs} | Dokładność testowa: {accuracy:.2f}%")

np.savez('wagi_modelu.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Wagi zapisane w 'wagi_modelu.npz'.")
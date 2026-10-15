import os
import idx2numpy
import numpy as np
import kagglehub

try:
    dane = np.load("model.npz")
except:
    print("Upewnij sie ze wygenerowales model poprzez wytrenuj.py")
    
weights_1 = dane["weights_1"]
weights_2 = dane["weights_2"]

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

x_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-images.idx3-ubyte"))
x_test = x_test.reshape(-1, 28 * 28)
y_test = idx2numpy.convert_from_file(os.path.join(path, "t10k-labels.idx1-ubyte"))

def relu(x):
    return (x >= 0) * x

seed = 884736743
rng = np.random.default_rng(seed)

index = np.random.randint(0, 9999)

layer_0 = x_test[index]

layer_1 = np.dot(layer_0, weights_1)

layer_1 = relu(layer_1)

dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)

layer_1 *= dropout_mask * 2

layer_2 = np.dot(layer_1, weights_2)

print("Poprawna odpowiedz: " + str(y_test[index]))
print("Model wybral: " + str(np.argmax(layer_2)))
print()
for i in range(len(layer_2)):
    print(str(i) + ": " + str(round(layer_2[i], 3)))

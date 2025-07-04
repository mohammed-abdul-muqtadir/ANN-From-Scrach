import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # for numerical stability
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Parameter initialization
def initialize_parameters(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# Forward pass
def forward_propagation(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return a1, a2, z1, z2

# Cross-entropy loss
def compute_loss(a2, y):
    m = y.shape[0]
    a2 = np.clip(a2, 1e-12, 1 - 1e-12)  # prevent log(0)
    loss = -np.sum(y * np.log(a2)) / m
    return loss

# Backward pass
def backward_propagation(x, y, a1, a2, z1, z2, w2):
    m = y.shape[0]
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, w2.T) * relu_derivative(z1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dw1, db1, dw2, db2

# Parameter update
def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2

# Training loop
def train_model(x, y, hidden_size=128, learning_rate=0.05, epochs=50):
    input_size = x.shape[1]
    output_size = y.shape[1]

    w1, b1, w2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        a1, a2, z1, z2 = forward_propagation(x, w1, b1, w2, b2)
        loss = compute_loss(a2, y)
        dw1, db1, dw2, db2 = backward_propagation(x, y, a1, a2, z1, z2, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return w1, b1, w2, b2

# Prediction
def predict(x, w1, b1, w2, b2):
    _, a2, _, _ = forward_propagation(x, w1, b1, w2, b2)
    return np.argmax(a2, axis=1)

# Accuracy calculation
def accuracy(x, y, w1, b1, w2, b2):
    preds = predict(x, w1, b1, w2, b2)
    true = np.argmax(y, axis=1)
    return np.mean(preds == true)

# Main function
def main():
    w1, b1, w2, b2 = train_model(x_train, y_train, hidden_size=128, learning_rate=0.05, epochs=50)
    print("Train Accuracy:", accuracy(x_train, y_train, w1, b1, w2, b2))
    print("Test Accuracy:", accuracy(x_test, y_test, w1, b1, w2, b2))

if __name__ == "__main__":
    main()

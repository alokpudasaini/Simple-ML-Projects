import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST CSV (e.g., mnist_train_100.csv)
data = pd.read_csv("data.csv").values
X = data[:, 1:] / 255.0  # Normalize pixels
y_raw = data[:, 0]

# One-hot encode labels
y = np.zeros((len(y_raw), 10))
for i, label in enumerate(y_raw):
    y[i, label] = 1

# Network architecture
input_size = 784
hidden_size = 64
output_size = 10

# Initialize weights
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def softmax(z):
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def compute_cost(y_hat, y):
    m = y.shape[0]
    return -np.sum(y * np.log(y_hat + 1e-8)) / m

alpha = 0.5
epochs = 1000
m = X.shape[0]
cost_history = []

for i in range(epochs):
    # Forward propagation
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)  # predictions

    # Cost
    cost = compute_cost(A2, y)
    cost_history.append(cost)

    # Backpropagation
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Gradient descent update
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {cost:.4f}")

# Predict labels
predictions = np.argmax(A2, axis=1)
true_labels = np.argmax(y, axis=1)
accuracy = np.mean(predictions == true_labels)

print(f"Training accuracy: {accuracy * 100:.2f}%")

plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost (Cross-Entropy Loss)")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

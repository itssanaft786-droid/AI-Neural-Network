import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Dummy Digit Data
# -----------------------------
np.random.seed(42)

num_samples = 1000
num_features = 64     # 8x8 image flattened
num_classes = 10

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, num_classes, num_samples)

# One-hot encoding
Y = np.zeros((num_samples, num_classes))
Y[np.arange(num_samples), y] = 1

# Train-test split
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
y_test_labels = y[split:]

# -----------------------------
# 2. Initialize Parameters
# -----------------------------
input_size = num_features
hidden_size = 32
output_size = num_classes

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 100
loss_history = []

# -----------------------------
# 3. Activation Functions
# -----------------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# -----------------------------
# 4. Loss Function
# -----------------------------
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# -----------------------------
# 5. Training Loop
# -----------------------------
for epoch in range(epochs):
    # Forward Propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)

    # Loss
    loss = cross_entropy_loss(Y_train, y_pred)
    loss_history.append(loss)

    # Backpropagation
    dz2 = y_pred - Y_train
    dW2 = np.dot(a1.T, dz2) / X_train.shape[0]
    db2 = np.mean(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X_train.T, dz1) / X_train.shape[0]
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # Update Parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# 6. Model Evaluation
# -----------------------------
z1_test = np.dot(X_test, W1) + b1
a1_test = relu(z1_test)
z2_test = np.dot(a1_test, W2) + b2
test_predictions = softmax(z2_test)

predicted_labels = np.argmax(test_predictions, axis=1)
accuracy = np.mean(predicted_labels == y_test_labels)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# 7. Visualization
# -----------------------------
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

import numpy as np

# Hyperparameters
learning_rate = 0.1
epochs = 100000

def ReLu(x):
    return np.maximum(0, x)

def sigmoid(x):
    # Outputs between zero and one
    return 1 / (1 + np.exp(-x))

def log_loss(predicted, actual):
    buffer = 1e-15
    # Notice "lossfn" below; we place a tiny buffer above 0 and below 1 to prevent log(0)
    predicted = np.clip(predicted, buffer, 1 - buffer)
    # Batch size
    B = actual.shape[0]
    lossfn = -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)) / B
    return lossfn

# Define params globally so "L" can be accessed when locating final output
params = {}

def init_params(layer_sizes):
    # Range begins at 1; no parameters for input layer (also ensures "layer_sizes[l-1]" exists)
    for l in range(1, len(layer_sizes)):
        # (F_in x F_out) shape is convention due to each input sample being structured as a row vector
        params[f"W{l}"] = np.random.randn(layer_sizes[l-1], layer_sizes[l])
        # Matrix broadcasting rules allows column vector to be added to output, no matter the shape
        params[f"b{l}"] = np.zeros((layer_sizes[l],))
    return params

# Define "cache" globally so final output can be accessed when computing loss function
cache = {}

def forward_pass(input_matrix, params):
    A = input_matrix
    cache["A0"] = A
    # Each layer has weight and bias matrix; therefore, floor division by 2, then add 1 to account for range clipping
    L = len(params) // 2
    for l in range(1, L + 1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        # (Output x Weight) order is convention due to each input sample being structured as a row vector
        Z  = A @ W + b
        cache[f"Z{l}"] = Z
        # Choose activation function based on layer (final should use binary classifier)
        # Reassign to prevent the same "A" value from being reused in multiple layers
        if l == L:
            A = sigmoid(Z)
        else:
            A = ReLu(Z)
        cache[f"A{l}"] = A
    return A

# Dataset goes here (w/ batch size defined globally for ease of access)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Gradient cache defined globally for ease of access
gradient_cache = {}

def backward_pass(params, cache, labels):
    batch_size = labels.shape[0]
    L = len(params) // 2
    last_output = cache[f"A{L}"]
    # Start with output layer (since it's the only layer that uses sigmoid rather than ReLu)
    # Derivative of "log-loss" with respect to Z_L
    dZ = last_output - labels

    for l in reversed(range(1, L+1)):
        A_prev = cache[f"A{l-1}"]
        W = params[f"W{l}"]

        # Derivate of ReLu with respect to weight
        dW = A_prev.T @ dZ / batch_size
        # Derivative of ReLu with respect to bias
        db = np.sum(dZ, axis=0) / batch_size

        gradient_cache[f"dW{l}"] = dW
        gradient_cache[f"db{l}"] = db

        if l > 1:
            Z_prev = cache[f"Z{l-1}"]
            # Derivative through activation function
            dA_prev = dZ @ W.T
            # Reassign dZ based on new position in network (using ReLu derivative this time)
            dZ = dA_prev * (Z_prev > 0)

def update_params():
    L = len(params) // 2
    for l in range(1, L+1):
        params[f"W{l}"] -= learning_rate * gradient_cache[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * gradient_cache[f"db{l}"]

def train():
    init_params([2,8,1])
    L = len(params) // 2
    for epoch in range(epochs):
        forward_pass(inputs, params)
        loss = log_loss(cache[f"A{L}"], labels)
        backward_pass(params, cache, labels)
        update_params()
        if epoch % 10000 == 0:
            print(f"Loss: {loss}")

def predict():
    predicted_probabilities = forward_pass(inputs, params)
    predictions = (predicted_probabilities > 0.5).astype(int)
    print(f"\nPrediction:\n{predictions}\n")

train()
predict()

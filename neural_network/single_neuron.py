import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets.utilities import *
from tqdm import tqdm


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


def artificial_neuron(X_train, y_train, learning_rate=0.01, n_iter=1000):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        # activation Train
        A_train = model(X_train, W, b)

        if i % 10 == 0:
            # Train
            train_loss.append(log_loss(A_train, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

        # mise a jour
        dW, db = gradients(A_train, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.legend()

    plt.show()

    return (W, b)


X_train, y_train = load_data()
X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()

W, b = artificial_neuron(X_train_reshape, y_train,
                         learning_rate=0.01, n_iter=10000)
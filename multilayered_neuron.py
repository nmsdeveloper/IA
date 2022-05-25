import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets.utilities import *
from tqdm import tqdm


def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' +
                   str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


def forward_propagation(X, parametres):
    activations = {'A0': X}

    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres[f"W{c}"].dot(
            activations[f"A{c - 1}"]) + parametres[f"b{c}"]
        activations[f"A{c}"] = 1 / (1 + np.exp(-Z))

    return activations


def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def back_propagation(y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations[f"A{C}"] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients[f"dW{c}"] = 1 / m * np.dot(dZ, activations[f"A{c - 1}"].T)
        gradients[f"db{c}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres[f"W{c}"].T, dZ) * \
                activations[f"A{c - 1}"] * (1 - activations[f"A{c - 1}"])

    return gradients


def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres[f"W{c}"] = parametres[f"W{c}"] - \
            learning_rate * gradients[f"dW{c}"]
        parametres[f"b{c}"] = parametres[f"b{c}"] - \
            learning_rate * gradients[f"db{c}"]

    return parametres


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations[f"A{C}"]
    return Af >= 0.5


def deep_neural_network(X, y, hidden_layers=(100, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 1), learning_rate=0.001, n_iter=28000):
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations[f"A{C}"]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
        y_pred = predict(X, parametres)
        training_history[i, 1] = (
            accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(training_history[:, 0], label='train loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(training_history[:, 1], label='train acc')
    # plt.legend()
    # plt.show()

    return training_history


X_train, y_train = load_data()

X_train = X_train.T
y_train = y_train.T

X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

training_history = deep_neural_network(
    X_train_reshape, y_train, learning_rate=0.1, n_iter=100)

print(training_history)
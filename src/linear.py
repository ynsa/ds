import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def train_perceptron(X_train, y_train, X_test, y_test):
    """Train perceptron on provided data and return accuracy."""

    clf = Perceptron()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def get_data(filepath):
    """Get X and y from csv file."""
    data = np.genfromtxt(filepath, delimiter=',')
    return data[:, 1:], data[:, 0]


def main():
    X_train, y_train = get_data('../data/perceptron-train.csv')
    X_test, y_test = get_data('../data/perceptron-test.csv')
    accuracy = train_perceptron(X_train, y_train, X_test, y_test)
    print(accuracy)

    # normalized
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    accuracy_normalized = train_perceptron(X_train_scaled, y_train,
                                           X_test_scaled, y_test)
    print(accuracy_normalized)

    with open('../results/linear.txt', 'w') as f:
        f.write(f'{accuracy_normalized - accuracy:.3f}')


if __name__ == '__main__':
    main()

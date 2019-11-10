import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def log_loss(w, x, y):
    return np.log(e(w, x, y))


def log_loss_l(w, x, y, l, C):
    return sum([log_loss(w, x[i], y[i]) for i in range(l)]) / l \
           + 0.5 * C * sum([w_**2 for w_ in w])


def sigmoid(w, x, y):
    return pow(e(w, x, y), -1)


def e(w, x, y):
    return 1 + np.exp(-y * sum([w[i] * x[i] for i in range(len(x))]))


def w_recalculate(w, k, l, y, x, w_n, C):
    sum_ = [
        y[i] * x[i][w_n] *
        (1 - sigmoid(w, x[i], y[i]))
        for i in range(l)
    ]
    return w[w_n] + k * (1 / l) * sum(sum_) - k * C * w[w_n]


def main(X, y, C, T: int = 10000, k: float = 0.1, eps: float = pow(10, -5)):
    l = len(X)
    aucs = []

    for c in C:
        w = [0] * X.shape[1]
        w_old = [0] * X.shape[1]
        for t in range(T):
            for i in range(len(w)):
                w[i] = w_recalculate(w_old, k, l, y, X, i, c)

            loss = log_loss_l(w, X, y, l, c)
            print(f'{t}: {loss}')

            if np.sqrt(sum([(w[i] - w_old[i])**2 for i in range(X.shape[1])])) \
                    < eps:
                break

            w_old = w.copy()

        y_score = [pow(1 + np.exp(- w[0] * X[i][0] - w[1] * X[i][1]), -1)
                   for i in range(len(X))]

        auc = roc_auc_score(y, y_score)
        print(f'\nC={c}:\tauc={auc}\n\n')
        aucs.append(auc)

        fpr, tpr, thresholds = roc_curve(y, y_score)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    with open('../results/logistic.txt', 'w') as f:
        f.write(f'{aucs[0]:.3f} {aucs[1]:.3f}')


if __name__ == '__main__':
    data = np.genfromtxt('../data/data-logistic.csv', delimiter=',')
    y = data[:, 0]
    X = data[:, 1:]
    C = [0, 10]
    main(X, y, C)

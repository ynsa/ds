import numpy as np
from sklearn.svm import SVC


if __name__ == '__main__':
    data = np.genfromtxt('../data/svm-data.csv', delimiter=',')
    y = data[:, 0]
    X = data[:, 1:]

    clf = SVC(C=100000, random_state=241, kernel='linear')
    clf.fit(X, y)
    print(clf.support_)

    with open('../results/svm.txt', 'w') as f:
        f.write(' '.join([str(x + 1) for x in clf.support_]))

import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def binary_linear_standart(X, y):
    """
    Бинарный линейный классификатор из sklearn
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Линейный классификатор sklearn:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def binary_linear_from_scratch(X, y):
    """
    Собственный бинарный линейный классификатор
    """

    class Binary_classifier_from_scratch:
        """
        Самописный бинарный линейный классификатор
        """

        def __init__(self):
            self.w = None
            self.b = None

        def model(self, x):
            return 1 if (np.dot(self.w, x) >= self.b) else 0

        def predict(self, X):
            pred = []
            for x in X:
                result = self.model(x)
                pred.append(result)
            return np.array(pred)

        def fit(self, X, y, lr=0.1):
            self.w = np.ones(X.shape[1])
            self.b = 0
            for x, y in zip(X, y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # classifier = Binary_classifier_from_scratch()
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print('Собственный бинарный линейный классификатор:')
    # print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')

    # зависимость accuracy от learning rate
    accs = []
    lrs = []

    for lr in np.linspace(0, 1, num=100):
        classifier = Binary_classifier_from_scratch()
        classifier.fit(X_train, y_train, lr=lr)
        lrs.append(lr)
        accs.append(metrics.accuracy_score(y_test, classifier.predict(X_test)))

    plt.plot(lrs, accs)
    plt.title('Зависимость accuracy от learning rate')
    plt.xlabel('learning rate')
    plt.ylabel('accuracy')
    plt.savefig('accuracy vs learning rate')
    plt.show()


def binary_linear_from_scratch_sgd(X, y):
    """
    линейный классификатор обучаемый градиентным спуском
    """

    class Binary_classifier_from_scratch:
        """
        Самописный бинарный линейный классификатор
        """

        def __init__(self):
            self.w = None
            self.b = None

        def a_xw(self, x):
            """
            sign (sum wi*xi)
            """
            return 1 if (np.dot(self.w, x) >= 0) else -1

        def loss_function(self, x, y):
            """
            S(M) = 2/(1+exp(M))
            """
            return 2 / (1 + math.exp(y * self.a_xw(x)))

        def loss_function_derivative(self, x, y):
            return -2 * math.exp(y * self.a_xw(x)) / (1 + math.exp(y * self.a_xw(x)) ** 2)

        def predict(self, X):
            pred = []
            for x in X:
                result = self.a_xw(x)
                pred.append(result)
            return np.array(pred)

        def fit(self, X, y, lr=0.1, forget=0.1):
            self.w = np.random.rand(X.shape[1])
            self.b = np.random.rand(1)[0]

            q_func = 0
            # обучение линейного классификатора с sgd
            while True:
                w_old = self.w
                # оценка функционала
                for x, y in zip(X, y):
                    q_func += self.loss_function(x, y)
                q_func /= X.shape[0]
                # выбор объекта из тренировочной выборки
                obj = np.random.choice([x for x in range(X.shape[0])])
                # вычислить потерю
                error = self.loss_function(X[obj], y)
                # градиентный шаг
                self.w -= lr * self.loss_function_derivative(X[obj], y)
                # оценка функционала
                q_new = forget * error + (1 - forget) * q_func
                # q или веса не сошлись
                if (q_func - q_new) < 1e-7 or np.linalg.norm(self.w - w_old) < 1e-7:
                    break

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = Binary_classifier_from_scratch()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Собственный бинарный линейный классификатор:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def main():
    """
    Задание:
    Вручную реализовать бинарный линейный классификатор и сравнить со стандартным из sklearn
    """

    # Бинарная классификация - медицинские данные по анализу рака груди
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target

    # бинарный линейный классификатор из sklearn
    binary_linear_standart(X_cancer, y_cancer)

    # самописный бинарный линейный классификатор
    binary_linear_from_scratch_sgd(X_cancer, y_cancer)


if __name__ == "__main__":
    main()

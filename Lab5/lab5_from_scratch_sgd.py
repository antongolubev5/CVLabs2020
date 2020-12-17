import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Binary_classifier_from_scratch:
    """
    Самописный линейный классификатор с sgd
    """

    def __init__(self, eta=0.01, forget=0.1, n_iter=10, shuffle=True, random_state=None):
        self.cost_ = [1.]
        self.eta = eta
        self.n_iter = n_iter
        self.forget = forget
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.steps_tmp = 0

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        while True:
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for x, target in zip(X, y):
                cost.append(self._update_weights(x, target))
            # Оценка функционала Q с коэффициентом <<momentum (forget)>>
            # В качестве самого функционала выступает усредненное значение функции потерь + учитываем <<забывание>>
            # формула из презентации
            avg_cost = self.forget * (sum(cost) / len(y)) + (1 - self.forget) * self.cost_[-1]
            self.steps_tmp += 1
            # Остановка на основе нормы изменения весов работает плохо, тк классификатор сходится за несколько шагов, не
            # обучившись. --- np.linalg.norm(self.w_ - w_tmp) < 1e-9
            # Критерий останова по функционалу или числу шагов
            if avg_cost < 1e-4 or self.steps_tmp > self.n_iter:
                # print('Число шагов = ', str(self.steps_tmp))
                break
        return self

    def _shuffle(self, X, y):
        """
        перемешивание экземпляров
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        инициализация весов с помощью распределения гаусса
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=1., size=1 + m)
        self.w_initialized = True

    def _update_weights(self, x, target):
        """
        Обновления весов сразу для квадратичной функции потерь, то есть
        градиенты уже взяты руками и для каждого веса записаны
        """
        output = self.activation_function(self.net_input(x))
        error = (target - output)
        self.w_[1:] += self.eta * x.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation_function(self, X):
        """
        Функция активации S(M) = 2 / (1+exp(-M))
        Если использовать функцию S(M) = M F_1 мера выше на 5%
        """
        return 2. / (1 + np.exp(-X))
        # return X

    def predict(self, X):
        """
        предсказание модели
        """
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)


def binary_linear_from_scratch(X_train, y_train, X_test, y_test):
    """
    Самописный бинарный линейный классификатор с sgd
    """
    clf = Binary_classifier_from_scratch(eta=0.0005, forget=0.7, n_iter=150)
    clf.fit(X_train, y_train)

    print('\nСамописный линейный классификатор:')
    print('F1 мера на тренировочных данных = %f' % metrics.f1_score(y_train, clf.predict(X_train)))
    print('F1 мера на тестовых данных = %f\n' % metrics.f1_score(y_test, clf.predict(X_test)))


def binary_linear_standart(X_train, y_train, X_test, y_test):
    """
    Бинарный линейный классификатор из sklearn
    """
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Линейный классификатор sklearn:')
    print('F1 мера = {:.2f}'.format(metrics.f1_score(y_test, y_pred)) + '\n')


def main():
    # загрузка и стандартизация данных
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer)
    transformer = StandardScaler().fit(X_cancer)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    # самописный линейный классификатор с sgd
    binary_linear_from_scratch(X_train, y_train, X_test, y_test)

    # вариант из sklearn
    binary_linear_standart(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

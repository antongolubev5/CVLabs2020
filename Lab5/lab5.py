import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def show_image(img, title, gray):
    """
    отрисовка изображения
    :param img: массив точек
    :param title: название картинки
    :param gray: True = отрисовка в оттенках серого
    """
    if gray:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img)

    plt.title(title)
    plt.savefig(title)
    plt.show()


def log_reg_multi(X, y):
    """
    Логистическая регрессия для множества классов
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Логистическая регрессия для многоклассовой задачи:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def log_reg_binary(X, y):
    """
    Бинарная логистическая регрессия
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Логистическая регрессия для бинарной задачи:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def binary_svm(X, y):
    """
    Бинарный svm
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('SVM для бинарной задачи:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def binary_linear(X, y):
    """
    Бинарный линейный классификатор
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Линейный классификатор для бинарной задачи:')
    print('Accuracy = {:.2f}'.format(metrics.accuracy_score(y_test, y_pred)) + '\n')


def main():
    """
    Задание:
    Вручную реализовать бинарный линейный классификатор
    """

    # Многоклассовый датасет - цифры от 0 до 9
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target

    for i in range(3):
        show_image(digits.data[i].reshape(8, 8), 'Цифра ' + str(y_digits[i]), True)

    # мультиклассовая логрег
    log_reg_multi(X_digits, y_digits)

    # Бинарная классификация - медицинские данные по анализу рака груди
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target

    # бинарная логрег
    log_reg_binary(X_cancer, y_cancer)

    # бинарный SVM
    binary_svm(X_cancer, y_cancer)

    # бинарный линейный классификатор
    binary_linear(X_cancer, y_cancer)


if __name__ == "__main__":
    main()

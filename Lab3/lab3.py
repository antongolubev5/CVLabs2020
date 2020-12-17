import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from collections import Counter
from numpy import dot
from numpy.linalg import norm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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


def plot_hist(labels):
    """
    построение гистограммы из массива меток
    :param labels:  массив меток
    """
    cntr = Counter(labels)
    plt.bar(cntr.keys(), cntr.values())
    plt.title('Распределение классов по датасету')
    plt.xlabel('Номер класса')
    plt.ylabel('Число элементов')
    plt.savefig('classes_distribution')
    plt.show()


def load_data_and_some_eda():
    """
    загрузка данных и небольшой разведочный анализ
    """
    # встроенный датасет с лицами знаменитостями
    # параметр - мин. объём класса (20 для умеренного числа классов для будущей кластеризации)
    data = fetch_lfw_people(min_faces_per_person=20)

    # проверка изображений
    for i in range(3):
        show_image(data.images[i], data.target_names[data.target[i]], gray=True)

    y = data.target
    X = data.images
    X = X / 255  # нормировка для устойчивости

    print('Количество классов = ', str(len(np.unique(y))))
    print('Размер изображения = ', str(X[0].shape))

    # распределение классов, датасет сбалансирован
    plot_hist(y)
    return X, y


def accuracy_vs_number_of_components(X_train, X_test, y_train, y_test):
    neighbors = []
    accuracies = []

    for i in range(1, 100):
        pca = PCA(n_components=i).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_pca, y_train)
        neighbors.append(i)
        accuracies.append(knn.score(X_test_pca, y_test))

    plt.plot(neighbors, accuracies)
    plt.title('Точность классификации KNN (k=3) vs число главных компонент PCA')
    plt.xlabel('Число компонент')
    plt.ylabel('Точность')
    plt.savefig('accuracy vs number of principal components')
    plt.show()


def cosine_similarity(a, b):
    """
    значение косинусной близости между двумя векторами
    """
    return dot(a, b) / (norm(a) * norm(b))


def main():
    """
    Задание:
    Поиск похожего лица с помощью PCA. Алгоритм должен уметь находить PCA у картинки с лицом
    (отдельно детектор лица реализовывать не нужно) и возвращать меру схожести между разными лицами.
    По этой мере схожести будем оценивать, тот же ли человек на фото. Можно взять любой сет данных небольшого размера.
    """

    X, y = load_data_and_some_eda()

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # тренировка алгоритма pca
    pca = PCA(n_components=100).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # кластеризация методом knn с числом соседей = 3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    print('точность knn с числом соседей = 3 равна ' + str(np.round(knn.score(X_test_pca, y_test), 3)))

    # построение зависимость между числом выделяемых главных компонент и точностью классификации
    # (немного подождать)
    accuracy_vs_number_of_components(X_train, X_test, y_train, y_test)

    # исходя из гистограммы примем значение числа компонент равным 90

    # сравнение схожести фотографией людей в пределах одного класса и между разными классами
    pca = PCA(n_components=90).fit(X_train)
    X_train_pca_best = pca.transform(X_train)
    X_test_pca_best = pca.transform(X_test)

    # подсчет среднего значения косинусной близости между pca-векторами в пределах одного класса
    # найдем все элементы класса 10
    idxs = np.where(y_test == 10)[0]
    pca_vectors = X_test_pca_best[idxs]

    # подсчет среднего значения косиусной близости между элементами одного класса
    avg_cosine_similarity = 0
    cnt = 0
    for i in range(pca_vectors.shape[0]):
        for j in range(pca_vectors.shape[0]):
            avg_cosine_similarity += cosine_similarity(pca_vectors[i], pca_vectors[j])
            cnt += 1

    print('Среднее значение косинусной близости между элементами одного класса = ',
          str(np.round(avg_cosine_similarity / cnt, 3)))

    # подсчет среднего значения косиусной близости между элементами разных классов
    # найдем произвольный элемент класса 11 и сравним его со всеми элементами класса 10
    another_class_element = X_test_pca_best[np.where(y_test == 11)[0]][0]

    avg_cosine_similarity = 0
    cnt = 0
    for i in range(pca_vectors.shape[0]):
        avg_cosine_similarity += cosine_similarity(pca_vectors[i], another_class_element)
        cnt += 1

    print('Среднее значение косинусной близости между элементами разных классов = ',
          str(np.round(avg_cosine_similarity / cnt, 3)))

    print('Вывод: отличить изображения с помощью данного метода представляется возможным')


if __name__ == "__main__":
    main()

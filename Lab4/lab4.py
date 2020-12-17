import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from tabulate import tabulate


def tokenizer(text):
    """
    токенизатор с удалением шума из текста
    """

    def drop_url(text):
        return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub('emoji', text)

    def drop_email(text):
        return re.sub('[\w\.-]+@[\w\.-]+', '', text)

    def drop_hash(text):
        return re.sub('(?<=#)\w+', '', text)

    def drop_mention(text):
        return re.sub('(?<=@)\w+', '', text)

    def drop_phone_number(text):
        return re.sub('\b\d{10}\b', '', text)

    def rep(text):
        grp = text.group(0)
        if len(grp) > 1:
            return grp[0:2]

    def unique_char(rep, sentence):
        return re.sub(r'(\w)\1+', rep, sentence)

    def drop_dates(text):
        return re.sub('\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b', 'date', text)

    def clean_text(text):
        """Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers."""
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def text_preprocessing(text):
        """
        Cleaning and parsing the text.

        """
        for foo in [drop_url, remove_emoji, drop_email, drop_mention, drop_phone_number, drop_phone_number]:
            text = foo(text)
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        nopunc = clean_text(text)
        tokenized_text = tokenizer.tokenize(nopunc)
        # remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
        return ' '.join(tokenized_text)

    return text_preprocessing(text)


def k_means_clusterization(X, y):
    """
    кластеризация методом k-средних
    """
    return KMeans(n_clusters=3).fit_predict(X)


def spectral_clusterization(X, y):
    """
    спектральная кластеризация
    """
    return SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0).fit_predict(X)


def calculate_metrics(y_true, y_pred_k_means, y_pred_spectral):
    """
    подсчет различных метрик для кластеризации
    """
    results = pd.DataFrame(
        columns=['Метод', 'Rand Index', 'Mutual Info Index', 'Гомогенность', 'Полнота', 'V-мера',
                 'Fowlkes Mallows Index'])

    results = results.append(pd.Series(
        ['K-means',
         str(np.round(metrics.adjusted_rand_score(y_true, y_pred_k_means), 2)),
         str(np.round(metrics.adjusted_mutual_info_score(y_true, y_pred_k_means), 2)),
         str(np.round(metrics.homogeneity_score(y_true, y_pred_k_means), 2)),
         str(np.round(metrics.completeness_score(y_true, y_pred_k_means), 2)),
         str(np.round(metrics.v_measure_score(y_true, y_pred_k_means), 2)),
         str(np.round(metrics.fowlkes_mallows_score(y_true, y_pred_k_means), 2))],
        index=results.columns), ignore_index=True).append(pd.Series(
        ['Спектральная',
         str(np.round(metrics.adjusted_rand_score(y_true, y_pred_spectral), 2)),
         str(np.round(metrics.adjusted_mutual_info_score(y_true, y_pred_spectral), 2)),
         str(np.round(metrics.homogeneity_score(y_true, y_pred_spectral), 2)),
         str(np.round(metrics.completeness_score(y_true, y_pred_spectral), 2)),
         str(np.round(metrics.v_measure_score(y_true, y_pred_spectral), 2)),
         str(np.round(metrics.fowlkes_mallows_score(y_true, y_pred_spectral), 2))],
        index=results.columns), ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(tabulate(results, headers='keys', tablefmt='psql'))


def load_and_clean_data():
    """
    датасет с 217k вопросами  из разных категорий к телевикторине
    оставим только 1000 вопросов из 5 приведенных ниже категорий
    удаляем ненужные столбцы
    """
    data = pd.read_csv('JEOPARDY_CSV.csv')
    data = data[[' Category', ' Question']]
    data = data[data[' Category'].isin(['FOOD', 'OPERA', 'LANGUAGES'])]
    data.rename(columns={' Category': 'label', ' Question': 'text'}, inplace=True)
    labels = data['label'].values
    categories = {'FOOD': 1, 'OPERA': 2, 'LANGUAGES': 3}
    data['label'] = data['label'].map(categories)
    return data, labels


def preprocessing(data):
    """
    токенизация, кодирование категорий, кодирование текстов с помощью tf-idf (dimension=300)
    """
    data['text'] = data['text'].apply(tokenizer)
    y = data['label'].values
    X = data['text'].values

    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(X)
    print('Количество текстов =  ' + str(X.shape[0]))
    print('Количество классов =  ' + str(len(np.unique(y))))
    print('Размерность tf-idf представления каждого текста = ' + str(X.shape[1]))
    X = X.toarray()
    return X, y


def draw_pca_decomposition(X, labels):
    """
    отображение классов с помощью pca
    """
    pca = PCA(n_components=2).fit(X)
    X_pca = pca.transform(X)
    data_for_pca = pd.DataFrame({'x': X_pca[:, 0], 'y': X_pca[:, 1], 'label': labels})

    sns.scatterplot(x="x", y="y", hue="label", data=data_for_pca, palette=['blue', 'green', 'red'])
    plt.title('Тексты после pca-декомпозиции в пространстве с dim=2')
    plt.xlabel('Главная компонента №1')
    plt.ylabel('Главная компонента №2')
    plt.savefig('after pca dim=2')
    plt.show()


def main():
    """
    Задание:
    Кластеризация двумя методами: спектральная и k-means и сравнение по различным метрикам из sklearn
    """

    # загрузка и предобработка данных
    data, labels = load_and_clean_data()

    # препроцессинг текстов + tf-idf
    X, y = preprocessing(data)

    # смотрим на pca-декомпозицию в 2-мерное пространство (3 класса вполне неплохо разделяются)
    draw_pca_decomposition(X, labels)

    # k_means кластеризация
    y_pred_k_means = k_means_clusterization(X, y)

    # спектральная кластеризация
    y_pred_spectral = spectral_clusterization(X, y)

    # подсчет метрик
    calculate_metrics(y, y_pred_k_means, y_pred_spectral)

    print('Вывод: Метод K-means проявляет себя лучше спектральной кластеризации на этих данных')


if __name__ == "__main__":
    main()

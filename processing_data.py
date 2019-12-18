import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from stopword_remover import delete_words_x
from sklearn.externals import joblib


CV = CountVectorizer()
X_name = 'Пример текста'
Y_name = 'Класс'
MAP = {0: 'VACATION-REQUEST', 1: 'SALARY-REQUEST', 2: 'SICK-LEAVE-REPORT', 3: 'OTHER'}
P_MIN = 0.41


def loading_data(file):
    df = pd.read_csv(file, delimiter=',', encoding="utf-8").astype(str)
    x = df[X_name].values
    y = df[Y_name].values
    return x, y


def preprocess_data(file):
    x, y = loading_data(file)
    x_train = preprocess_x(x)
    y_train = preprocess_y(y)
    return x_train, y_train


def preprocess_x(x):
    delete_words_x(x)
    x_train = CV.fit_transform(x).toarray()
    joblib.dump(CV, 'CV.pkl')
    return x_train


def preprocess_y(y):
    y_train = list()
    for i, y_tmp in enumerate(y):
        for number in MAP:
            if y_tmp == MAP[number]:
                y_train.append(number)
    return y_train


def test(cv, file):
    x, y = loading_data(file)
    return query(cv, x), y


def query(cv, x):
    delete_words_x(x)
    x_test = cv.transform(x).toarray()
    return x_test


def convert_y_to_str(y_probability):
    max_prob = max(y_probability)
    if max_prob < P_MIN:
        return MAP[3]
    return MAP[list(y_probability).index(max_prob)]


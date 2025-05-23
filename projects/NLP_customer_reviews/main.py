

#
#
# UTILITY FUNCTION

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier


def _get_current_folder() -> str:
    # import os path namespace
    import os.path
    # get the absolute path from the current file we're in
    absolute_path = os.path.abspath(__file__)
    # get the folder of the absolute path
    return os.path.dirname(absolute_path)

# UTILITY FUNCTION


csv_filepath = f"{_get_current_folder()}/customer-reviews.csv"


dataset = pd.read_csv(csv_filepath, delimiter='\t', quoting=3)

total_rows = dataset.shape[0]
print(f"total_rows {total_rows}")

corpus: list[str] = []
output: list[int] = []


# Define language model
# -> must run -> python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

for i in range(0, total_rows):

    review = dataset['Review'][i]

    doc = nlp(review)

    tokens_filtrered = []

    for token in doc:
        if (token.is_stop and (token.lemma_ != 'not')) or token.is_punct:
            continue

        tokens_filtrered.append(token.lemma_)

    review = " ".join(tokens_filtrered)

    # skip empty reviews
    if (len(review) == 0):
        continue

    corpus.append(review)
    output.append(dataset['Liked'][i])

print("len(corpus)", len(corpus))


X = np.array(corpus)
y = np.array(output)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

bag_of_word_size = 1500

cv = TfidfVectorizer(max_features=bag_of_word_size, ngram_range=(1, 2))
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

print("X.shape", X.shape)


class my_Random_Forest:
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=10, criterion='entropy', random_state=0)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_Decision_Tree:
    def __init__(self):
        self.classifier = DecisionTreeClassifier(
            criterion='entropy', random_state=0)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_Gaussian_NB:
    def __init__(self):
        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_SVC:
    def __init__(self):
        self.classifier = SVC(kernel='rbf', random_state=0)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_Logistic_Regression:
    def __init__(self):
        self.classifier = LogisticRegression(random_state=0, C=1)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_KNeighbors:
    def __init__(self):
        self.classifier = KNeighborsClassifier(
            n_neighbors=5, metric='minkowski', p=2)
        self.classifier.fit(X_train, y_train)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_ann:
    def __init__(self):
        model_filepath = f"{_get_current_folder()}/my-model.keras"

        model_file = Path(model_filepath)

        if model_file.exists() and model_file.is_file():

            ann = tf.keras.models.load_model(model_filepath)

        else:

            print("model file was not found")
            print("training new model")

            ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
            ann.add(tf.keras.layers.Dropout(rate=0.3))
            ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
            ann.add(tf.keras.layers.Dropout(rate=0.3))
            ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

            ann.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=['accuracy'])

            ann.fit(X_train, y_train, batch_size=32, epochs=100)

            ann.save(model_filepath)

        self.ann = ann

    def predict(self, X_input):
        return self.ann.predict(X_input, verbose=0).reshape(-1,)


class my_Catboost:
    def __init__(self):
        self.classifier = CatBoostClassifier()
        self.classifier.fit(X_train, y_train, verbose=0)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_prediction_class:
    name: str
    y_pred: np.ndarray
    accuracy_score: float

    def __init__(self, name: str, predictor):
        self.name = name
        self.predictor = predictor
        self.y_pred = self.predictor.predict(X_test) > 0.5
        self.accuracy_score = accuracy_score(y_test, self.y_pred)


all_predictions: list[my_prediction_class] = []


my_predictor = my_Random_Forest()
all_predictions.append(my_prediction_class(
    "random forest     ", my_predictor))
my_predictor = my_Decision_Tree()
all_predictions.append(my_prediction_class(
    "decision tree     ", my_predictor))
my_predictor = my_Gaussian_NB()
all_predictions.append(my_prediction_class(
    "GaussianNB        ", my_predictor))

my_predictor = my_Logistic_Regression()
all_predictions.append(my_prediction_class(
    "LogisticRegression", my_predictor))

my_predictor = my_KNeighbors()
all_predictions.append(my_prediction_class(
    "KNeighbors        ", my_predictor))

my_predictor = my_ann()
all_predictions.append(my_prediction_class(
    "ann               ", my_predictor))

my_predictor = my_SVC()
all_predictions.append(my_prediction_class(
    "svc               ", my_predictor))

my_predictor = my_Catboost()
all_predictions.append(my_prediction_class(
    "Catboost          ", my_predictor))


print("unsorted")
for curr_values in all_predictions:
    print("->", curr_values.name, curr_values.accuracy_score)

# sort by score


def my_sort_func_by_score(values: my_prediction_class):
    return values.accuracy_score


all_predictions.sort(reverse=True, key=my_sort_func_by_score)


print("sorted")
for curr_values in all_predictions:
    print("->", curr_values.name, curr_values.accuracy_score)


def _predict_my_sentence(sentence: str, expected: bool):

    # convert to the bag of words we trained
    X_raw = cv.transform([sentence]).toarray()

    # predict
    print(f"########################")
    print(f" -> custom sentence -> '{sentence}'")
    print(f" -> we expect: {expected}")
    for value in all_predictions:
        y_raw = value.predictor.predict(X_raw)
        y_raw = (y_raw >= 0.5)

        if y_raw[0] == expected:
            print(
                f" ---> prediction '{value.name}' say {y_raw[0]} -----> SUCCESS")
        else:
            print(
                f" ---> prediction '{value.name}' say {y_raw[0]} -----> FAILURE")


all_custom_sentences: list[tuple[str, bool]] = []
all_custom_sentences.append(("the food was amazing", True))
all_custom_sentences.append(("the food was great", True))
all_custom_sentences.append(("the food was good", True))
all_custom_sentences.append(("slow service but great taste", True))
all_custom_sentences.append(("the food was bad", False))
all_custom_sentences.append(("I'll will not order from them again", False))
all_custom_sentences.append(("avoid that place", False))

for sentence in all_custom_sentences:
    _predict_my_sentence(sentence[0], sentence[1])

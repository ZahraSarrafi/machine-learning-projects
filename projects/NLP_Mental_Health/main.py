from xgboost import XGBClassifier
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


#
#
#
#
# load and process original dataset
# UTILITY FUNCTION
csv_filepath = f"{_get_current_folder()}/Mental Health Dataset.csv"
my_csv_filepath = f"{_get_current_folder()}/my_Mental Health Dataset.csv"


my_csv_file = Path(my_csv_filepath)

if my_csv_file.exists() and my_csv_file.is_file():

    print("cached dataset file was found")
    print("reusing previously pre-processed dataset")

    my_df = pd.read_csv(my_csv_filepath)

else:

    print("cached dataset file was not found")
    print("pre-processing dataset")

    dataset = pd.read_csv(csv_filepath)
    # Removing missing values

    dataset = dataset.dropna(ignore_index=True)
    total_rows = dataset.shape[0]
    print(f"total_rows {total_rows}")

    corpus: list[str] = []
    output: list[int] = []

    # Define language model
    # -> must run -> python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    for i in range(0, total_rows):

        post = dataset['posts'][i]

        if i % 100 == 0:
            print(f"{i}/{total_rows}")

        if not isinstance(post, str):
            print(f"line {i} is skipped, not a string ")
            continue

        if len(post.split()) <= 15:
            print(f"line {i} is skipped, less than 15 words ")
            continue

        doc = nlp(post)

        tokens_filtrered = []

        for token in doc:
            if (token.is_stop and (token.lemma_ != 'not')) or token.is_punct:
                continue

            tokens_filtrered.append(token.lemma_)

        post = " ".join(tokens_filtrered)

        # skip empty reviews
        if (len(post) == 0):
            continue

        corpus.append(post)
        output.append(dataset['intensity'][i])

    print("len(corpus)", len(corpus))

    my_df = pd.DataFrame({
        'corpus': corpus,
        'output': output
    })

    my_df.to_csv(my_csv_filepath, index=False)

# load and process original dataset
#
#
#
#


X = my_df["corpus"]
y = my_df["output"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=0)

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


class my_xgboost:
    def __init__(self):

        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()

        # will encode [-2,-1,0,1] to [0,1,2,3]
        y_train2 = self.le.fit_transform(y_train)

        self.classifier = XGBClassifier()
        model_filepath = f"{_get_current_folder()}/my_xgboost.model"
        model_file = Path(model_filepath)

        if model_file.exists() and model_file.is_file():
            self.classifier.load_model(model_filepath)
        else:
            self.classifier.fit(X_train, y_train2)
            self.classifier.save_model(model_filepath)

    def predict(self, X_input):
        y_pred = self.classifier.predict(X_input)
        # will decode [0,1,2,3] to [-2,-1,0,1]
        return self.le.inverse_transform(y_pred)


class my_ann:
    def __init__(self):
        model_filepath = f"{_get_current_folder()}/my-model.keras"

        # tf.keras.config.set_floatx('float64')

        model_file = Path(model_filepath)
        from sklearn.preprocessing import OneHotEncoder
        self.encoder = OneHotEncoder()
        # must convert to numpy array, just to be sure
        y_train2 = self.encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test2 = self.encoder.transform(y_test.reshape(-1, 1)).toarray()

        if model_file.exists() and model_file.is_file():

            ann = tf.keras.models.load_model(model_filepath)

        else:

            print("model file was not found")
            print("training new model")

            ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(units=20, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.L2(0.001)))
            ann.add(tf.keras.layers.Dense(units=20, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.L2(0.001)))
            ann.add(tf.keras.layers.Dense(units=4, activation='sigmoid'))

            ann.compile(optimizer='adamw', loss='binary_crossentropy',
                        metrics=['accuracy'])

            early_stopping = tf.keras.callbacks.EarlyStopping(
                # monitor='val_loss',
                monitor='val_accuracy',
                # monitor='mean_squared_error',

                # how long with no progress do we insist?
                patience=300,

                restore_best_weights=True,
                verbose=1
            )

            ann.fit(
                X_train, y_train2,
                epochs=300,
                batch_size=32,
                validation_data=(X_test, y_test2),
                callbacks=[early_stopping],
                verbose=1
            )

            ann.save(model_filepath)

        self.ann = ann

    def predict(self, X_input):

        y_pred = self.ann.predict(X_input, verbose=0)
        y_pred = self.encoder.inverse_transform(y_pred)
        return y_pred


class my_Catboost:
    def __init__(self):
        self.classifier = CatBoostClassifier()
        model_filepath = f"{_get_current_folder()}/my_Catboost.model"

        model_file = Path(model_filepath)

        if model_file.exists() and model_file.is_file():
            self.classifier.load_model(model_filepath)
        else:
            self.classifier.fit(X_train, y_train, verbose=0)
            self.classifier.save_model(model_filepath)

    def predict(self, X_input):
        return self.classifier.predict(X_input)


class my_prediction_class:
    name: str
    y_pred: np.ndarray
    accuracy_score: float

    def __init__(self, name: str, predictor):
        self.name = name
        self.predictor = predictor
        self.y_pred = self.predictor.predict(X_test)
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

my_predictor = my_xgboost()
all_predictions.append(my_prediction_class(
    "xgboost           ", my_predictor))

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

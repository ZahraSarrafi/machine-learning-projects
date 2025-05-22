from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
from pathlib import Path


def _get_current_folder() -> str:
    # import os path namespace
    import os.path
    # get the absolute path from the current file we're in
    absolute_path = os.path.abspath(__file__)
    # get the folder of the absolute path
    return os.path.dirname(absolute_path)


csv_filepath = f"{_get_current_folder()}/Cancer_Data.Csv"
df = pandas.read_csv(csv_filepath)

print("# RAW DATAFRAME")
print(df)

X = df.iloc[:, 1:-1].values
print("X (inputs)")
print(X)

# will contains the values of the column 'class:Tumor type'
y = df.iloc[:, -1].values
y = (y > 3)

print("y (outputs)")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("X_train.shape:", X_train.shape)
print("X_train:", X_train)
print("X_test.shape:", X_test.shape)
print("X_test:", X_test)


def RandomForest_pred(X_input):

    classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_input)

    return y_pred


def DecisionTree_pred(X_input):
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_input)

    return y_pred


def GaussianNB_pred(X_input):

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_input)

    return y_pred


def SVC_pred(X_input):

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_input)

    return y_pred


def LogisticRegression_pred(X_input: np.ndarray) -> np.ndarray:

    classifier = LogisticRegression(random_state=0, C=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_input)

    return y_pred


def KNeighbors_pred(X_input):

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_input)

    return y_pred


def ann_prediction(X_input):

    model_filepath = f"{_get_current_folder()}/Tumor-type.keras"

    model_file = Path(model_filepath)

    if model_file.exists() and model_file.is_file():

        print("model file was found")
        print("reusing previously trained model")

        ann = tf.keras.models.load_model(model_filepath)

    else:

        print("model file was not found")
        print("training new model")

        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(
            units=6,  activation='relu'))
        ann.add(tf.keras.layers.Dropout(rate=0.3))
        ann.add(tf.keras.layers.Dense(
            units=6,  activation='relu'))
        ann.add(tf.keras.layers.Dropout(rate=0.3))
        ann.add(tf.keras.layers.Dense(
            units=1,  activation='sigmoid'))

        ann.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['accuracy'])

        ann.fit(X_train, y_train, batch_size=32, epochs=100)

        ann.save(model_filepath)

    y_pred = ann.predict(X_input, verbose=0)
    return y_pred


class my_prediction_class:
    name: str
    y_pred: np.ndarray
    accuracy_score: float

    def __init__(self, name: str, predict_func):
        self.name = name
        self.predict_func = predict_func
        self.y_pred = predict_func(X_test) > 0.5
        self.accuracy_score = accuracy_score(y_test, self.y_pred)


all_predictions: list[my_prediction_class] = []


all_predictions.append(my_prediction_class(
    "deep learning", ann_prediction))

all_predictions.append(my_prediction_class(
    "logistic regression", LogisticRegression_pred))

all_predictions.append(my_prediction_class(
    "k-nearest-neighbors", KNeighbors_pred))
all_predictions.append(my_prediction_class(
    "svc", SVC_pred))
all_predictions.append(my_prediction_class(
    "naive bayes", GaussianNB_pred))
all_predictions.append(my_prediction_class(
    "decision tree", DecisionTree_pred))
all_predictions.append(my_prediction_class(
    "random forest", RandomForest_pred))

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

best_pred = all_predictions[0]  # Highest value
worst_pred = all_predictions[-1]  # Lowest value


print(best_pred)
print(worst_pred)

# Does she have breast canser?


def my_function(
        Clump, uniformity_Cell_Size, uniformity_Cell_Shape, Marginal_Adhesion, Single_Epithelial_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses):

    # print(f"Does she with these features {Clump},{uniformity Cell Size},{Uniformity Cell Shape},{Marginal Adhesion},{Single Epithelial Cell Size}, {Bare Nuclei}, {Bland Chromatin}, {Normal Nucleoli}, {Mitoses} have breast cancer?")

    input_data = sc.transform([[Clump, uniformity_Cell_Size, uniformity_Cell_Shape, Marginal_Adhesion,
                              Single_Epithelial_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses]])

    print("Nural Network say:", ann_prediction(input_data))
    print("LogisticRegressionsay:", LogisticRegression_pred(input_data))
    print("NKNeighbors say:", KNeighbors_pred(input_data))
    print("SVC say:", SVC_pred(input_data))
    print("GaussianNB say:", GaussianNB_pred(input_data))
    print("DecisionTree say:", DecisionTree_pred(input_data))
    print("RandomForest say:", RandomForest_pred(input_data))


my_function(4, 1, 1, 3, 2, 1, 3, 1, 1)
my_function(8, 10, 10, 8, 7, 10, 9, 7, 1)

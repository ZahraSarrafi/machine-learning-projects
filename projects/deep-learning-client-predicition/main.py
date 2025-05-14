

#
#
# UTILITY FUNCTION

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# UTILITY FUNCTION
#
#


# to run in a terminal -> 'pip install pandas'

csv_filepath = f"{_get_current_folder()}/Social_Network_Ads.csv"


# df -> means "data frame"
df = pandas.read_csv(csv_filepath)


print()
print("#")
print("# RAW DATAFRAME")
print("#")
print()

print(df)


X = df.iloc[:, 1:-1].values
print("X (inputs)")
print(X)
y = df.iloc[:, -1].values  # will contains the values of the column 'Purchased'

print("y (outputs)")
print(y)

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

print("Gender is now encoded", X)


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

    model_filepath = f"{_get_current_folder()}/my-model.keras"

    model_file = Path(model_filepath)

    if model_file.exists() and model_file.is_file():

        print("model file was found")
        print("reusing previously trained model")

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

# Visualizing

# men and women purchase plot
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 1].min() - 10,
              stop=X_set[:, 1].max() + 10, step=5.0),
    np.arange(start=X_set[:, 2].min() - 10,
              stop=X_set[:, 2].max() + 10, step=5.0)
)

ones_array = np.ones(len(X1.ravel()))
zeros_array = np.zeros(len(X1.ravel()))

fig, (row1, row2) = plt.subplots(2, 2, figsize=(7, 7))

(ax1, ax2) = row1
(ax3, ax4) = row2

ax1.contourf(X1, X2, best_pred.predict_func(sc.transform(np.array([ones_array, X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
for i, j in enumerate(np.unique(y_set)):
    ax1.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
ax1.set_title(f'Best Did Purchase (for Men)\n"{best_pred.name}"', fontsize=8)
ax1.set_xlabel('Age')
ax1.set_ylabel('Estimated Salary')
ax1.legend()


ax2.contourf(X1, X2, best_pred.predict_func(sc.transform(np.array([zeros_array, X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
for i, j in enumerate(np.unique(y_set)):
    ax2.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
ax2.set_title(f'Best Did Purchase (for women)\n"{best_pred.name}"', fontsize=8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Estimated Salary')
ax2.legend()

ax3.contourf(X1, X2, worst_pred.predict_func(sc.transform(np.array([ones_array, X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
for i, j in enumerate(np.unique(y_set)):
    ax3.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
ax3.set_title(f'Worst Did Purchase (for men)\n"{worst_pred.name}"', fontsize=8)
ax3.set_xlabel('Age')
ax3.set_ylabel('Estimated Salary')
ax3.legend()

ax4.contourf(X1, X2, worst_pred.predict_func(sc.transform(np.array([zeros_array, X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
for i, j in enumerate(np.unique(y_set)):
    ax4.scatter(X_set[y_set == j, 1], X_set[y_set == j, 2],
                c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
ax4.set_title(
    f'Worst Did Purchase (for women)\n"{worst_pred.name}"', fontsize=8)
ax4.set_xlabel('Age')
ax4.set_ylabel('Estimated Salary')
ax4.legend()

plt.tight_layout()
plt.show(block=True)  # <- force the window to open and stay open

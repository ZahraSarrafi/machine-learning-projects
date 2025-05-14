

#
#
# UTILITY FUNCTION

from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas
import numpy as np


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


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann.add(tf.keras.layers.Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix", cm)
print(f" -> did not purchase prediction")
print(f"   -> correct   {cm[0][0]}")
print(f"   -> incorrect {cm[1][0]}")
print(f" -> did purchase prediction")
print(f"   -> correct   {cm[1][1]}")
print(f"   -> incorrect {cm[0][1]}")


print("accuracy_score", accuracy_score(y_test, y_pred))

# predict a 40yo Male earning 60k

# will the person purchase?
print(ann.predict(sc.transform([[1, 40, 60000]])) > 0.5)
# purchase probability?
print(ann.predict(sc.transform([[1, 40, 60000]])))

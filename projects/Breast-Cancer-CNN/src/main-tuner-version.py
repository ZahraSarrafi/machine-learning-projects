

from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import keras_tuner


def _get_current_folder() -> str:
    # import os path namespace
    import os.path
    # get the absolute path from the current file we're in
    absolute_path = os.path.abspath(__file__)
    # get the folder of the absolute path
    return os.path.dirname(absolute_path)


def _get_all_filepath_from_folder(folder_filepath: str) -> list[str]:
    all_filepath: list[str] = []
    for curr_file in os.listdir(folder_filepath):
        full_path = os.path.join(folder_filepath, curr_file)
        if os.path.exists(full_path):
            all_filepath.append(full_path)

    return all_filepath


train_benign = f"{_get_current_folder()}/../assets/ultrasound breast classification/train/benign"
train_malignant = f"{_get_current_folder()}/../assets/ultrasound breast classification/train/malignant"
test_benign = f"{_get_current_folder()}/../assets/ultrasound breast classification/val/benign"
test_malignant = f"{_get_current_folder()}/../assets/ultrasound breast classification/val/malignant"

# make a list of all the filepath for train+benign
all_train_benign = _get_all_filepath_from_folder(train_benign)
all_train_malignant = _get_all_filepath_from_folder(train_malignant)
all_test_benign = _get_all_filepath_from_folder(test_benign)
all_test_malignant = _get_all_filepath_from_folder(test_malignant)

print("all_train_benign    ", len(all_train_benign))
print("all_train_malignant ", len(all_train_malignant))
print("all_test_benign     ", len(all_test_benign))
print("all_test_malignant  ", len(all_test_malignant))

# we use "zip()" to merge 2 lists into 1 list of a tuple with 2 elements
column_train_benign = list(zip(all_train_benign, [0] * len(all_train_benign)))
column_train_malignant = list(
    zip(all_train_malignant, [1] * len(all_train_malignant)))
column_test_benign = list(zip(all_test_benign, [0] * len(all_test_benign)))
column_test_malignant = list(
    zip(all_test_malignant, [1] * len(all_test_malignant)))

# we use "list.extend()" to merge 2 lists into 1 list
column_train_benign.extend(column_train_malignant)
column_test_benign.extend(column_test_malignant)


# here we create the train/test dataframes
train_df = pd.DataFrame(column_train_benign, columns=[
                        'filepath', 'is_malignant'])
test_df = pd.DataFrame(column_test_benign, columns=[
                       'filepath', 'is_malignant'])

# here we shuffle dataframes
# -> otherwise all the benign are at the start and all the malignant at the end
# -> here we use random_state to keep it deterministic between each launch
train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=0).reset_index(drop=True)

print('train_df')
print(train_df.info())
print(train_df)

print('test_df')
print(test_df.info())
print(test_df)

#
#
#


img_square_size = 128

# will map the images filepath with their loaded images data


def _get_img_filepath_to_map(all_filepaths: list[str]) -> dict[str, np.ndarray]:

    total = len(all_filepaths)

    img_filepath_map = {}

    for index, curr_filepath in enumerate(all_filepaths):

        if index > 0 and index % 1000 == 0:
            print(
                f" -> loading images -> progress: {index}/{total} ({(index / total) * 100.0:.0f}%)")

        # load the image -> https://keras.io/api/data_loading/image/
        test_img = tf.keras.utils.load_img(
            curr_filepath,
            color_mode="rgb",
            target_size=(img_square_size, img_square_size),  # <- will resize
            # <- no aliasing when resized (not blurry)
            interpolation="nearest",
            keep_aspect_ratio=False,
        )

        # convert the colored image to gray scales
        img_data = tf.image.rgb_to_grayscale(test_img)

        # convert the image to an array that can be passed to a model
        img_data = tf.keras.utils.img_to_array(img_data)

        # save the loaded image data against it's filepath
        img_filepath_map[curr_filepath] = img_data

    return img_filepath_map


print(f"loading + mapping the image data against their filepath")
to_load = [
    ('all_train_benign', all_train_benign),
    ('all_train_malignant', all_train_malignant),
    ('all_test_benign', all_test_benign),
    ('all_test_malignant', all_test_malignant)
]
all_filepath_img_data_map: dict[str, np.ndarray] = {}
for list_name, curr_list in to_load:
    print(f"starting list: '{list_name}'")
    all_filepath_img_data_map |= _get_img_filepath_to_map(curr_list)
    print(f" ---> loading images done: {len(all_filepath_img_data_map)}")

print('all_filepath_img_data_map  ->', len(all_filepath_img_data_map))

#
# debug

# import matplotlib.pyplot as plt
# plt.imshow(all_filepath_img_data_map[all_train_benign[0]])
# plt.axis('off')
# plt.show()

# debug
#

# get the list of image data (train)
X_train = np.array(
    list(map(lambda filepath: all_filepath_img_data_map.get(filepath), train_df.iloc[:, 0])))
print('X_train.shape', X_train.shape)

# get the list of is_malignant values (train)
y_train = np.array(train_df.iloc[:, 1])
print('y_train.shape', y_train.shape)


# get the list of image data (test)
X_test = np.array(
    list(map(lambda x: all_filepath_img_data_map.get(x), test_df.iloc[:, 0])))
print('X_test.shape', X_test.shape)

# get the list of is_malignant values (test)
y_test = np.array(test_df.iloc[:, 1])
print('y_test.shape', y_test.shape)


def my_keras_code(
    units_feature_layers: int,
    total_feature_layers: int,
    activation_feature_layers: str,
    units_dense_layers: int,
    total_dense_layers: int,
    activation_dense_layers: str,
    optimizer: str,
    epoch: int,
) -> float:

    # Build model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((img_square_size, img_square_size, 1)))

    # add X feature layers
    for i in range(0, total_feature_layers):

        my_local_units = (i + 1) * units_feature_layers

        model.add(tf.keras.layers.Conv2D(my_local_units, kernel_size=(
            3, 3), activation=activation_feature_layers, padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), padding='same'))

    # now switch to one dense layer
    model.add(tf.keras.layers.Flatten())

    # add X dense layers
    for i in range(0, total_dense_layers):
        model.add(tf.keras.layers.Dense(
            units_dense_layers, activation=activation_dense_layers))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

    # output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    #
    #
    #

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',

        # here "patience" is the same as "epoch"
        # -> we're just after the 'restore_best_weights' feature
        patience=epoch,

        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        validation_data=(X_test, y_test),
        # callbacks=[early_stopping],
        callbacks=[early_stopping],
        epochs=epoch,
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred = (y_pred > 0.5)

    score = accuracy_score(y_test, y_pred)
    print("accuracy_score", score)

    # Return a single float as the objective value.
    return score  # higher is better


class MyTuner(keras_tuner.GridSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return my_keras_code(

            units_feature_layers=hp.Int("units_feature_layers", 32, 512, 32),
            total_feature_layers=hp.Int("total_feature_layers", 1, 6, 1),
            activation_feature_layers=hp.Choice(
                "activation_feature_layers", ["linear", "sigmoid", "relu", "tanh", "leaky_relu"]),

            units_dense_layers=hp.Int("units_dense_layers", 32, 512, 32),
            total_dense_layers=hp.Int("total_dense_layers", 1, 6, 1),
            activation_dense_layers=hp.Choice(
                "activation_dense_layers", ["linear", "sigmoid", "relu", "tanh", "leaky_relu"]),

            optimizer=hp.Choice(
                "optimizer", ["adam", "adadelta", "adamw", "adagrad"]),

            epoch=3,
        )


tuner = MyTuner(
    # important: this set it
    objective=keras_tuner.Objective("val_accuracy", direction="max"),
    max_trials=10000,
    executions_per_trial=2,
    # overwrite=True,
    # directory=f"{_get_current_folder()}/my_dir",
    # project_name="keep_code_separate",
)

tuner.search_space_summary()

tuner.search()

# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]

print("best_hp", best_hp.values)

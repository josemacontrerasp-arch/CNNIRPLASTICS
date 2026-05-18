# THIS IS HEAVILY BASED ON THIS EXAMPLE: https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py
# ALL CODE WAS UNDERSTOOD AND REWRITTEN BY ME EXCLUDING SOME FUNCTION CALLS

import os
import sys
import subprocess
import numpy as np

def main(k):

    dbs1_path = r"C:\Users\20242972\Downloads\FTIR_PLASTIC_c4.csv"
    dbs2_path = r"C:\Users\20242972\Downloads\FTIR_PLASTIC_c8.csv"
    data = []

    # Here we extract the data and drop all columns except the label, data(x), and data(y)

    import csv
    with open(dbs1_path, newline="") as dbs:
        dbs_reader = csv.reader(dbs)
        dbs_reader.__next__()
        for row in dbs_reader:
            # c4 dataset samples wavenumbers at twice the frequency of c8, so we skip every second data x and y pair. Wavenumbers match, I checked
            corrected_row = [row[0]]
            for i in range(6, len(row), 4):
                for k in [float(row[i]), float(row[i+1])]:
                    corrected_row.append(k)
            data.append(np.array(corrected_row))

    with open(dbs2_path, newline="") as dbs:
        dbs_reader = csv.reader(dbs)
        dbs_reader.__next__()
        for row in dbs_reader:
            ar = np.array([row[0]])
            ar = np.concatenate([ar, np.array(row[6:len(row) - 2])])
            data.append(ar)


    # Data exists in data array - next we ensure existing data is evenly spaced with regards to wavenumber
    # Data x (wavenumber) exists at indices <6 with step 2
    prev_avg_delta_x = 0
    for row in range(0, len(data), 10):
        prev_delta_x = (float(data[row][3]) - float(data[row][1]))
        for cell in range(5, len(data[row]), 2):
            new_delta_x = float(data[row][cell]) - float(data[row][cell-2])
            if abs(new_delta_x - prev_delta_x) > prev_delta_x * 0.02:
                raise ValueError(f"uh oh, data is not evenly spaced at row {row} and cell {cell}")
            prev_delta_x = new_delta_x
                
            
        if row == 0:
            prev_avg_delta_x = prev_delta_x
            continue

        if abs(prev_delta_x - prev_avg_delta_x) > prev_avg_delta_x * 0.02:
            raise ValueError(f"uh oh, data is not evenly spaced at row {row} compared to previous rows")
        prev_avg_delta_x = prev_delta_x
        

    # Now it is confirmed that data is evenly spaced and consistent throughout the whole array

    import sklearn.metrics as skm
    import tensorflow as tf
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from tensorflow import keras
    from tensorflow.keras import layers


    # Creating the CNN architecture - credit: https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py
    def cnn1d(shape, seed):
            np.random.seed(seed)
            if tf.__version__ == '1.14.0':
                tf.set_random_seed(seed)
            else:
                tf.random.set_seed(seed)
            inputs = layers.Input(shape)
            x = layers.Conv1D(64, 3, activation='relu')(inputs)
            x = layers.Conv1D(64, 3, activation='relu')(x)
            x = layers.MaxPool1D()(x)

            x = layers.Conv1D(64, 3, activation='relu')(x)
            x = layers.Conv1D(64, 3, activation='relu')(x)
            x = layers.MaxPool1D()(x)

            x = layers.Flatten()(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)

            outputs = layers.Dense(10, activation='softmax')(x)
            model = keras.Model(inputs, outputs, name="fcnn")
            return model

    """ 
    Next we convert the labels to integers according to:
    0: HDPE
    1: LDPE
    2: PP
    3: PS
    4: PVC
    5: PET
    """
    label_to_int = {
        'HDPE': 0,
        'LDPE': 1,
        'PP': 2,
        'PS': 3,
        'PVC': 4,
        'PET': 5
    }

    for i in range(len(data)):
        label = str(data[i][0])
        if label.startswith("HDPE"):
            label = "HDPE"
        elif label.startswith("LDPE"):
            label = "LDPE"
        elif label.startswith("PP"):
            label = "PP"
        elif label.startswith("PS"):
            label = "PS"
        elif label.startswith("PVC"):
            label = "PVC"
        elif label.startswith("PET"):
            label = "PET"
        data[i] = (label_to_int[label], np.array([data[i][k] for k in range(1, len(data[i]), 2)], dtype=np.float64), np.array([data[i][k] for k in range(2, len(data[i]), 2)], dtype=np.float64))


    # Then we normalize the absorbance data to a range between 0 and 1 and drop the x values since they are the same for all spectra
    absorbance_values = [(row[2] - np.min(row[2])) / (np.max(row[2]) - np.min(row[2])) for row in data]
    labels = np.array([row[0] for row in data])

    for i in range(len(absorbance_values)):
        if len(absorbance_values[i]) != 1868:
            raise ValueError(f"uh oh, data at row {i} has {len(absorbance_values[i])} points instead of 1868")

    # Now the shape is 6000, 1868

    # Shuffle the data
    absorbance_values = np.random.RandomState(0).permutation(np.array(absorbance_values))
    labels = np.random.RandomState(0).permutation(labels)

    # Add "channels" dimension for CNN - keras Conv1D expects a 3D input of (samples, timesteps, channels)
    absorbance_values = absorbance_values[..., np.newaxis]

    # Next we split into train, validation, and test sets

    # Using StratifiedKFold ensures that each class (type of plastic) is represented roughly equally in each fold
    # Since n_splits=5, we will have 5 folds, where one fold is used for testing and others for training in each iteration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in skf.split(absorbance_values, labels):
        absorbance_train_fold, absorbance_test = absorbance_values[train_index], absorbance_values[test_index]
        labels_train_fold, labels_test = labels[train_index], labels[test_index]
        
        # Then we split training and validation sets using train_test_split - I didn't look into how this specifically works but I trust
        # that it does the job. Gives 30% for validation and 70% for training of the remaining 80% after the test fold is taken out
        absorbance_train_fold, absorbance_valid_fold, labels_train_fold, labels_valid_fold = train_test_split(
            absorbance_train_fold, labels_train_fold, random_state=0, test_size=0.3)

        i += 1
        if i == k:
            print(f"This is fold {i}.")
            break

    # Finally we create and compile the CNN model and set up early stopping, which stops training if the model stop improving

    model = cnn1d(
        shape=(
            absorbance_train_fold.shape[1],
            absorbance_train_fold.shape[2]),
        seed=0)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min", restore_best_weights=True)

    # Then we train the model

    hist = model.fit(
        absorbance_train_fold,
        keras.utils.to_categorical(
            labels_train_fold,
            num_classes=6),
        validation_data=(
            absorbance_valid_fold,
            keras.utils.to_categorical(
                labels_valid_fold,
                num_classes=6)),
        epochs=1000,
        shuffle=True,
        verbose=0,
        batch_size=64,
        callbacks=[
            early_stopping_cb])

    # Create predictions
    y_pred = np.argmax(model.predict(absorbance_test), axis=1)

    # Compare to test labels
    print(skm.accuracy_score(labels_test, y_pred))

    # View history
    print(hist.history)

if __name__ == "__main__":
    k_vals = [1, 2, 3, 4, 5]
    for k in k_vals:
        main(k)
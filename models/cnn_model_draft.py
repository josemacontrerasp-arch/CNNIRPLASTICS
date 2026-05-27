# THIS IS HEAVILY BASED ON THIS EXAMPLE: https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py
# ALL CODE WAS UNDERSTOOD AND REWRITTEN BY ME EXCLUDING SOME FUNCTION CALLS

import csv
from pathlib import Path
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def load_and_preprocess():
    BASE_DIR = Path(__file__).resolve().parent.parent

    dbs1_path = BASE_DIR / "data" / "FTIR_PLASTIC_c4.csv"
    dbs2_path = BASE_DIR / "data" / "FTIR_PLASTIC_c8.csv"
    data = []

    # Extract each row, keeping only the label, wavenumber (x), and absorbance (y) columns
    with open(dbs1_path, newline="") as dbs:
        dbs_reader = csv.reader(dbs)
        dbs_reader.__next__()
        for row in dbs_reader:
            # c4 samples wavenumbers at twice the frequency of c8, so we take every other (wavenumber, absorbance) pair
            # step=4 skips the duplicate pair; wavenumber alignment with c8 was verified manually
            corrected_row = [row[0]]
            for i in range(6, len(row), 4):
                for val in [float(row[i]), float(row[i+1])]:
                    corrected_row.append(val)
            data.append(np.array(corrected_row))

    with open(dbs2_path, newline="") as dbs:
        dbs_reader = csv.reader(dbs)
        dbs_reader.__next__()
        for row in dbs_reader:
            ar = np.array([row[0]])
            ar = np.concatenate([ar, np.array(row[6:len(row) - 2])])
            data.append(ar)

    # Verify that wavenumber spacing is uniform within and across spectra
    # (sampled every 10th row for performance; assumes the rest are consistent)
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

    """
    Convert string labels to integers:
    0: HDPE  1: LDPE  2: PP  3: PS  4: PVC  5: PET
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
        # Each row becomes (label_int, wavenumbers[], absorbance[]) — odd indices are wavenumber, even are absorbance
        data[i] = (
            label_to_int[label],
            np.array([data[i][j] for j in range(1, len(data[i]), 2)], dtype=np.float64),
            np.array([data[i][j] for j in range(2, len(data[i]), 2)], dtype=np.float64)
        )

    # Normalise absorbance per spectrum to [0, 1] and drop wavenumber axis (identical across all spectra)
    absorbance_values = [(row[2] - np.min(row[2])) / (np.max(row[2]) - np.min(row[2])) for row in data]
    labels = np.array([row[0] for row in data])

    for i in range(len(absorbance_values)):
        if len(absorbance_values[i]) != 1868:
            raise ValueError(f"uh oh, data at row {i} has {len(absorbance_values[i])} points instead of 1868")

    # Shape is now (6000, 1868)

    # Shuffle with a fixed seed for reproducibility
    absorbance_values = np.random.RandomState(0).permutation(np.array(absorbance_values))
    labels = np.random.RandomState(0).permutation(labels)

    # Add channel dimension — Conv1D expects (samples, timesteps, channels)
    absorbance_values = absorbance_values[..., np.newaxis]

    return absorbance_values, labels


def main(k, absorbance_values, labels):

    # Two conv blocks (each: Conv1D → Conv1D → MaxPool) followed by three dense layers with dropout
    # Credit: https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py
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

        outputs = layers.Dense(6, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name="fcnn")
        return model

    # Select the k-th fold as test set; remaining 80% is split 70/30 into train/validation
    # StratifiedKFold preserves class proportions across folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in skf.split(absorbance_values, labels):
        absorbance_train_fold, absorbance_test = absorbance_values[train_index], absorbance_values[test_index]
        labels_train_fold, labels_test = labels[train_index], labels[test_index]

        absorbance_train_fold, absorbance_valid_fold, labels_train_fold, labels_valid_fold = train_test_split(
            absorbance_train_fold, labels_train_fold, random_state=0, test_size=0.3)

        i += 1
        if i == k:
            print(f"This is fold {i}.")
            break

    model = cnn1d(
        shape=(absorbance_train_fold.shape[1], absorbance_train_fold.shape[2]),
        seed=0)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # SparseCategoricalCrossentropy accepts integer labels directly — no one-hot encoding needed
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min", restore_best_weights=True)

    hist = model.fit(
        absorbance_train_fold,
        labels_train_fold,
        validation_data=(absorbance_valid_fold, labels_valid_fold),
        epochs=1000,
        shuffle=True,
        verbose=0,
        batch_size=64,
        callbacks=[early_stopping_cb])

    y_pred = np.argmax(model.predict(absorbance_test), axis=1)

    print(f"Fold {k} accuracy: {skm.accuracy_score(labels_test, y_pred):.4f}")
    print(hist.history)

    save_dir = Path("output")
    save_dir.mkdir(exist_ok=True)
    model.save(save_dir / f"fold_{k}.keras")
    print(f"Model saved to {save_dir / f'fold_{k}.keras'}")


if __name__ == "__main__":
    absorbance_values, labels = load_and_preprocess()
    for k in [1]:
        main(k, absorbance_values, labels) #, 2, 3, 4, 5]: I will run each fold separately to avoid GPU memory issues if you have big boy computer
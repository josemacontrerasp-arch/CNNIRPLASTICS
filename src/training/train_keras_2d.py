import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from src.training.train_keras_1d import load_and_preprocess as load_1d
from src.models.gaf_2d_pipeline import load_gaf_dataset, prepare_gaf_dataset, save_gaf_dataset

# Full 1868×1868 GAF is too large; resize to this square resolution before training
GAF_SIZE = 64


def load_and_preprocess():
    target_shape = (GAF_SIZE, GAF_SIZE)
    try:
        return load_gaf_dataset(target_shape)
    except FileNotFoundError:
        print(f"No cached GAF dataset found — computing from scratch (this takes a few minutes)...")
        absorbance_values, labels = load_1d()
        spectra = np.squeeze(absorbance_values, axis=-1)
        gaf_images, labels = prepare_gaf_dataset(
            max_samples=None,
            target_shape=target_shape,
            spectra=spectra,
            labels=labels,
        )
        save_gaf_dataset(gaf_images, labels, target_shape=target_shape)
        return gaf_images, labels


def main(k, gaf_images, labels):

    # Two conv blocks (each: Conv2D → Conv2D → MaxPool2D) followed by three dense layers with dropout
    # Mirrors the 1D architecture in train_keras_1d.py, extended to 2D GAF images
    def cnn2d(shape, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        inputs = layers.Input(shape)

        x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(6, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name="cnn2d")
        return model

    # Select the k-th fold as test set; remaining 80% is split 70/30 into train/validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    i = 0
    for train_index, test_index in skf.split(gaf_images, labels):
        gaf_train_fold, gaf_test = gaf_images[train_index], gaf_images[test_index]
        labels_train_fold, labels_test = labels[train_index], labels[test_index]

        gaf_train_fold, gaf_valid_fold, labels_train_fold, labels_valid_fold = train_test_split(
            gaf_train_fold, labels_train_fold, random_state=0, test_size=0.3)

        i += 1
        if i == k:
            print(f"This is fold {i}.")
            break

    model = cnn2d(
        shape=(gaf_train_fold.shape[1], gaf_train_fold.shape[2], gaf_train_fold.shape[3]),
        seed=0)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min", restore_best_weights=True)

    hist = model.fit(
        gaf_train_fold,
        labels_train_fold,
        validation_data=(gaf_valid_fold, labels_valid_fold),
        epochs=1000,
        shuffle=True,
        verbose=0,
        batch_size=64,
        callbacks=[early_stopping_cb])

    y_pred = np.argmax(model.predict(gaf_test), axis=1)

    print(f"Fold {k} accuracy: {skm.accuracy_score(labels_test, y_pred):.4f}")
    print(hist.history)

    save_dir = Path("output/models_2d")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / f"fold_{k}.keras")
    print(f"Model saved to {save_dir / f'fold_{k}.keras'}")


if __name__ == "__main__":
    gaf_images, labels = load_and_preprocess()
    for k in [1, 2, 3, 4, 5]:
        main(k, gaf_images, labels)

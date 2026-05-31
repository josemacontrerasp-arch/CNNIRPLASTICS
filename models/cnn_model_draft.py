# THIS IS HEAVILY BASED ON THIS EXAMPLE: https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py
# ALL CODE WAS UNDERSTOOD AND REWRITTEN BY ME EXCLUDING SOME FUNCTION CALLS
#
# This file now contains ONLY the CNN model itself, split into three reusable
# functions:
#
#   create_model(input_shape, ...)  -> builds and compiles a fresh 1-D CNN
#   train_model(model, X, y, ...)   -> trains it (internal validation split +
#                                       early stopping) and returns the History
#   test_model(model, X)            -> returns (y_pred, y_proba)
#
# Data extraction, alignment, and preprocessing are NOT done here anymore -- the
# pipeline in run_pipeline.py handles those (format_data.py + preprocess.py) and
# feeds this module ready-to-train arrays.  Cross-validation (the StratifiedKFold
# split) is also driven by the caller, so these functions stay model-only and can
# be reused by any fold / data scope.

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape, n_classes=6, seed=0, learning_rate=0.0001):
    """Build and compile the 1-D CNN.

    Two conv blocks (each: Conv1D -> Conv1D -> MaxPool) followed by three dense
    layers with dropout.  Architecture/credit:
    https://github.com/zavalab/ML/blob/master/CNN_Plastic/code/train.py

    Parameters
    ----------
    input_shape : tuple
        (timesteps, channels) for the Conv1D input, e.g. (600, 1).
    n_classes : int
        Number of output classes (softmax units).
    seed : int
        Seed applied to numpy and TensorFlow for reproducible weight init.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    keras.Model
        Compiled model using sparse categorical cross-entropy (integer labels,
        no one-hot encoding needed).
    """
    np.random.seed(seed)
    if tf.__version__ == '1.14.0':
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)

    inputs = layers.Input(input_shape)
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

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="fcnn")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # SparseCategoricalCrossentropy accepts integer labels directly -- no one-hot needed
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["acc"])
    return model


def train_model(
    model,
    X_train,
    y_train,
    val_size=0.3,
    epochs=1000,
    batch_size=64,
    patience=100,
    seed=0,
    verbose=0,
):
    """Train a compiled model on one training fold.

    The training fold is split internally into train/validation (default 70/30)
    so EarlyStopping can monitor val_loss and restore the best weights.  This
    mirrors the original training code; only the surrounding extraction and
    k-fold logic moved out into run_pipeline.py.

    Parameters
    ----------
    model : keras.Model
        A compiled model from create_model().
    X_train, y_train : np.ndarray
        Training-fold features (samples, timesteps, channels) and integer labels.
    val_size : float
        Fraction of the training fold held out for validation / early stopping.
    epochs, batch_size, patience : int
        Standard fit / EarlyStopping hyperparameters.
    seed : int
        Seed for the internal train/validation split.
    verbose : int
        Keras fit verbosity.

    Returns
    -------
    keras.callbacks.History
        The training history (history.history holds loss/acc curves).
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, random_state=seed, test_size=val_size)

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min",
        restore_best_weights=True)

    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        shuffle=True,
        verbose=verbose,
        batch_size=batch_size,
        callbacks=[early_stopping_cb])
    return hist


def test_model(model, X_test):
    """Run a trained model on a test fold.

    Returns predictions in a format compatible with the comparison function in
    run_pipeline.py: integer class predictions plus the raw class probabilities.

    Parameters
    ----------
    model : keras.Model
        A trained model.
    X_test : np.ndarray
        Test features (samples, timesteps, channels).

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples,)
        Integer predicted class indices (argmax over the softmax output).
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Raw softmax probabilities, kept for ROC/threshold analysis later.
    """
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba

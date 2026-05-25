import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    keras = None
    layers = None


from src.utils.gaf_transform import stack_gaf_channels, transform_batch

# Default save location for the preprocessed GAF dataset
_DEFAULT_SAVE_DIR = Path("data/processed")


def _load_normalized_spectra():
    from src.training.train_keras_1d import load_and_preprocess

    absorbance_values, labels = load_and_preprocess()
    spectra = np.squeeze(absorbance_values, axis=-1)
    return spectra, labels


def prepare_gaf_dataset(max_samples=64, target_shape=None, spectra=None, labels=None):
    """
    Load the current normalized 1D spectra and convert them to GAF images.

    Parameters
    ----------
    max_samples:
        Limit the number of samples converted. Default is 64 so the
        2D pipeline can be tested without exhausting memory.
    target_shape:
        Optional resize target, e.g. (128, 128). When omitted, the
        full GAF matrix shape is kept.
    spectra:
        Optional preloaded 1D spectral batch. If omitted, the function
        will attempt to load the current dataset through the existing
        1D loader.
    labels:
        Optional labels matching the provided spectra.
    """
    if spectra is None and labels is None:
        spectra, labels = _load_normalized_spectra()
    elif spectra is None or labels is None:
        raise ValueError("spectra and labels must be provided together")

    spectra = np.asarray(spectra)
    labels = np.asarray(labels)
    if spectra.ndim == 3 and spectra.shape[-1] == 1:
        spectra = np.squeeze(spectra, axis=-1)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array of shape (samples, timesteps)")

    if max_samples is not None:
        spectra = spectra[:max_samples]
        labels = labels[:max_samples]

    gasf_batch, gadf_batch = transform_batch(spectra)
    gaf_images = stack_gaf_channels(gasf_batch, gadf_batch)

    if target_shape is not None:
        gaf_images = resize_gaf_batch(gaf_images, target_shape)

    return gaf_images.astype(np.float32), labels


def resize_gaf_batch(gaf_images, target_shape):
    """
    Resize a batch of GAF images to a smaller spatial size.

    Parameters
    ----------
    gaf_images:
        Array with shape (samples, height, width, channels).
    target_shape:
        Tuple of (height, width) used to downsample the GAF images.
    """
    if tf is None:
        raise RuntimeError(
            "TensorFlow is required for resize_gaf_batch. Install it or leave target_shape=None."
        )

    gaf_images = tf.convert_to_tensor(gaf_images, dtype=tf.float32)
    resized = tf.image.resize(gaf_images, target_shape, method="bilinear")
    return resized.numpy()


def save_gaf_dataset(gaf_images, labels, target_shape, save_dir=None):
    """
    Save a GAF dataset to a compressed .npz file in the data folder.

    The filename encodes the spatial resolution, e.g. gaf_64x64.npz,
    so different resolutions can coexist without overwriting each other.

    Parameters
    ----------
    gaf_images:
        Array with shape (samples, height, width, channels).
    labels:
        Integer label array with shape (samples,).
    target_shape:
        (height, width) tuple used when generating the images — embedded
        in the filename for traceability.
    save_dir:
        Directory to write the file into. Defaults to data/processed/.
    """
    save_dir = Path(save_dir) if save_dir is not None else _DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    h, w = target_shape
    out_path = save_dir / f"gaf_{h}x{w}.npz"
    np.savez_compressed(out_path, gaf_images=gaf_images, labels=labels)
    print(f"GAF dataset saved to {out_path}  shape={gaf_images.shape}")
    return out_path


def load_gaf_dataset(target_shape, save_dir=None):
    """
    Load a previously saved GAF dataset from the data folder.

    Parameters
    ----------
    target_shape:
        (height, width) tuple — must match the resolution the file was
        saved with.
    save_dir:
        Directory to search. Defaults to data/processed/.

    Returns
    -------
    gaf_images, labels  or raises FileNotFoundError.
    """
    save_dir = Path(save_dir) if save_dir is not None else _DEFAULT_SAVE_DIR
    h, w = target_shape
    path = save_dir / f"gaf_{h}x{w}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No saved GAF dataset found at {path}. Run gaf_2d_pipeline.py to generate it.")
    data = np.load(path)
    gaf_images = data["gaf_images"]
    labels = data["labels"]
    print(f"GAF dataset loaded from {path}  shape={gaf_images.shape}")
    return gaf_images, labels


def build_2d_cnn(input_shape, num_classes=6, seed=0):
    """
    Build a small 2D CNN for GAF images.
    """
    if keras is None or layers is None or tf is None:
        raise RuntimeError(
            "TensorFlow is required to build the 2D CNN. Install TensorFlow to use build_2d_cnn."
        )

    np.random.seed(seed)
    tf.random.set_seed(seed)

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="gaf_cnn")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save GAF dataset to data/processed/")
    parser.add_argument("--size", type=int, default=64,
                        help="Square GAF image resolution (default: 64)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override the output directory (default: data/processed/)")
    args = parser.parse_args()

    target_shape = (args.size, args.size)
    print(f"Building GAF dataset at {args.size}x{args.size} — this may take a few minutes...")
    gaf_images, labels = prepare_gaf_dataset(
        max_samples=None,
        target_shape=target_shape,
    )
    save_gaf_dataset(gaf_images, labels, target_shape=target_shape, save_dir=args.save_dir)

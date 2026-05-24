import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    tf = None
    keras = None
    layers = None


from models.gaf_transform import stack_gaf_channels, transform_batch


def _load_normalized_spectra():
    from models.cnn_model_draft import load_and_preprocess

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

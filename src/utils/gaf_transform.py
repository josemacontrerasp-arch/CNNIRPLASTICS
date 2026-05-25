import numpy as np


def min_max_scale(x):
    """
    Scale a 1D array to [0, 1] using min-max normalization.
    """
    x = np.asarray(x, dtype=np.float64)
    min_x = np.min(x)
    max_x = np.max(x)
    if max_x - min_x == 0:
        return np.zeros_like(x)
    return (x - min_x) / (max_x - min_x)


def compute_phi(x_scaled):
    """
    Compute the angular vector (phi) using arccos for a [0, 1] scaled vector.
    """
    # Clip to [0, 1] to avoid invalid values due to floating point errors
    x_scaled = np.clip(x_scaled, 0, 1)
    return np.arccos(x_scaled)


def compute_gasf(phi):
    """
    Compute the Gramian Angular Summation Field (GASF) matrix.
    GASF[i, j] = cos(phi[i] + phi[j])
    """
    phi = np.asarray(phi)
    return np.cos(phi[:, None] + phi[None, :])


def compute_gadf(phi):
    """
    Compute the Gramian Angular Difference Field (GADF) matrix.
    GADF[i, j] = sin(phi[i] - phi[j])
    """
    phi = np.asarray(phi)
    return np.sin(phi[:, None] - phi[None, :])


def gaf_transform(x):
    """
    Given a 1D IR vector, return (GASF, GADF) matrices.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        if x.shape[-1] == 1:
            x = np.squeeze(x, axis=-1)
        else:
            raise ValueError("gaf_transform expects a 1D IR vector")

    x_scaled = min_max_scale(x)
    phi = compute_phi(x_scaled)
    gasf = compute_gasf(phi)
    gadf = compute_gadf(phi)
    return gasf, gadf


def transform_batch(x_batch):
    """
    Transform a batch of 1D IR vectors into GASF/GADF matrices.

    Parameters
    ----------
    x_batch:
        Array with shape (samples, timesteps) or (samples, timesteps, 1).

    Returns
    -------
    gasf_batch, gadf_batch:
        Arrays with shape (samples, timesteps, timesteps).
    """
    x_batch = np.asarray(x_batch, dtype=np.float64)
    if x_batch.ndim == 3 and x_batch.shape[-1] == 1:
        x_batch = np.squeeze(x_batch, axis=-1)
    if x_batch.ndim != 2:
        raise ValueError("transform_batch expects a 2D batch of spectra")

    gasf_batch = np.empty((x_batch.shape[0], x_batch.shape[1], x_batch.shape[1]), dtype=np.float32)
    gadf_batch = np.empty((x_batch.shape[0], x_batch.shape[1], x_batch.shape[1]), dtype=np.float32)

    for i, sample in enumerate(x_batch):
        gasf, gadf = gaf_transform(sample)
        gasf_batch[i] = gasf.astype(np.float32)
        gadf_batch[i] = gadf.astype(np.float32)

    return gasf_batch, gadf_batch


def stack_gaf_channels(gasf_batch, gadf_batch):
    """
    Stack GASF and GADF matrices into a single channel dimension.

    Returns
    -------
    Array with shape (samples, height, width, 2).
    """
    gasf_batch = np.asarray(gasf_batch, dtype=np.float32)
    gadf_batch = np.asarray(gadf_batch, dtype=np.float32)

    if gasf_batch.shape != gadf_batch.shape:
        raise ValueError("GASF and GADF batches must have the same shape")

    return np.stack([gasf_batch, gadf_batch], axis=-1)

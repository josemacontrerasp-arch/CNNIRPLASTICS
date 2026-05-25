import numpy as np

def apply_horizontal_shift(spectrum, shift_limit=5):
    """
    Applies a random translation shift along the x-axis (wavenumber/frequency).
    Mimics real-world environmental shifts like tautomerism or solvent variations.

    Args:
        spectrum (np.ndarray): 1D array representing the spectrum vector.
        shift_limit (int): Maximum absolute shift in indices.

    Returns:
        np.ndarray: Shifted spectrum with boundary edge padding.
    """
    shift = np.random.randint(-shift_limit, shift_limit + 1)

    if shift == 0:
        return spectrum

    shifted_spectrum = np.zeros_like(spectrum)

    if shift > 0:
        # Shift right: Pad the start with the first value
        shifted_spectrum[shift:] = spectrum[:-shift]
        shifted_spectrum[:shift] = spectrum[0]
    else:
        # Shift left: Pad the end with the last value
        shift = abs(shift)
        shifted_spectrum[:-shift] = spectrum[shift:]
        shifted_spectrum[-shift:] = spectrum[-1]

    return shifted_spectrum

class SpectralAugmentor:
    """
    Wrapper for chemometric augmentations.
    """
    def __init__(self, shift_limit=5):
        self.shift_limit = shift_limit

    def __call__(self, spectrum):
        return apply_horizontal_shift(spectrum, self.shift_limit)

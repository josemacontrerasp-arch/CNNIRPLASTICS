"""
Composable, per-spectrum, leakage-free spectral preprocessing.

Every transform here operates row-wise on X (n_spectra x n_wavenumbers) using
ONLY each spectrum's own values -- no statistics are learned from the training
set -- so the whole pipeline is safe to apply outside a cross-validation loop.

Pipeline order (PreprocessConfig drives the toggles):
    smooth (Savitzky-Golay)
    -> baseline_correct (AsLS / arPLS)
    -> derivative (Savitzky-Golay)
    -> normalize (none | minmax | snv | l2)
    -> region_select (wavenumber window)

The default config is raw (everything off), so it doubles as the control.

Note on NaNs: Jesse's loader emits NaN for wavenumbers outside a spectrum's
measured range (almost entirely OpenSpecy). We zero-fill per spectrum at
pipeline entry -- this is itself a per-spectrum, leakage-free operation.

Usage:
    from preprocess import preprocess, PreprocessConfig, snv, baseline_correct
    Xp = preprocess(X, PreprocessConfig(baseline="asls", normalize="snv"))
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

try:
    from pybaselines import Baseline as _PybaselinesBaseline
    _HAS_PYBASELINES = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PYBASELINES = False

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Normalization (per-spectrum)
# ---------------------------------------------------------------------------

def snv(X: np.ndarray) -> np.ndarray:
    """Per-spectrum Standard Normal Variate: (x - mean) / std for each row.

    Vectorized over rows. Rows with std == 0 (flat spectra) keep std = 1.0 to
    avoid division by zero. After SNV every non-flat row has ~0 mean and ~1 std.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < _EPS, 1.0, std)
    out = (X - mean) / std

    # Sanity check: non-flat rows are standardized.
    finite = np.isfinite(out).all(axis=1)
    nonflat = finite & (X.std(axis=1) >= _EPS)
    if nonflat.any():
        assert np.allclose(out[nonflat].mean(axis=1), 0.0, atol=1e-6), \
            "SNV output rows are not ~zero mean"
        assert np.allclose(out[nonflat].std(axis=1), 1.0, atol=1e-6), \
            "SNV output rows are not ~unit std"
    return out


def minmax(X: np.ndarray) -> np.ndarray:
    """Per-spectrum min-max scaling to [0, 1]. Flat rows map to all zeros."""
    X = np.asarray(X, dtype=float)
    mn = X.min(axis=1, keepdims=True)
    rng = X.max(axis=1, keepdims=True) - mn
    rng = np.where(rng < _EPS, 1.0, rng)
    return (X - mn) / rng


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Per-spectrum L2 (unit-norm) scaling. Zero-norm rows are left unchanged."""
    X = np.asarray(X, dtype=float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm < _EPS, 1.0, norm)
    return X / norm


def normalize(X: np.ndarray, method: str = "none") -> np.ndarray:
    if method == "none":
        return np.asarray(X, dtype=float)
    if method == "minmax":
        return minmax(X)
    if method == "snv":
        return snv(X)
    if method == "l2":
        return l2_normalize(X)
    raise ValueError(f"unknown normalize method '{method}'")


# ---------------------------------------------------------------------------
# Baseline correction (Eilers' Asymmetric Least Squares + arPLS)
# ---------------------------------------------------------------------------

def _difference_matrix(n: int, order: int = 2) -> sparse.csc_matrix:
    """Sparse second-order difference operator D (shape (n-order, n))."""
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


def _asls_scipy(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    """scipy.sparse AsLS fallback (Eilers & Boelens 2005).

    Minimizes a second-difference smoothness penalty (lam) under an asymmetric
    reweighting that pushes the baseline below peaks (weight p above, 1-p below).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    D = _difference_matrix(n, 2)
    DtD = lam * (D.T @ D)
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0, format="csc")
        z = spsolve((W + DtD).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def _arpls_scipy(y: np.ndarray, lam: float, niter: int) -> np.ndarray:
    """scipy.sparse arPLS fallback (Baek et al. 2015).

    Adaptive variant: the asymmetry weight auto-tunes from the residual
    distribution below the baseline, so no fixed p is needed.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    D = _difference_matrix(n, 2)
    DtD = lam * (D.T @ D)
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = sparse.diags(w, 0, format="csc")
        z = spsolve((W + DtD).tocsc(), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        m, s = dn.mean(), dn.std()
        w_new = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * s - m)) / (s + _EPS)))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + _EPS) < 1e-3:
            w = w_new
            break
        w = w_new
    return z


def asls_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
    method: str = "asls",
) -> np.ndarray:
    """Estimate the baseline of a single spectrum.

    Uses pybaselines if installed, else a scipy.sparse fallback.
    method="asls" -> fixed-asymmetry Eilers AsLS.
    method="arpls" -> adaptive arPLS (auto-tunes asymmetry; ignores p).
    """
    if method not in ("asls", "arpls"):
        raise ValueError(f"unknown baseline method '{method}'")

    if _HAS_PYBASELINES:
        fitter = _PybaselinesBaseline()
        if method == "asls":
            bl, _ = fitter.asls(y, lam=lam, p=p, max_iter=niter)
        else:
            bl, _ = fitter.arpls(y, lam=lam, max_iter=niter)
        return bl

    if method == "asls":
        return _asls_scipy(y, lam=lam, p=p, niter=niter)
    return _arpls_scipy(y, lam=lam, niter=niter)


def baseline_correct(
    X: np.ndarray,
    method: str = "asls",
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """Return X minus the per-row estimated baseline.

    Sanity check: after correction the per-row baseline floor sits near zero
    (the corrected signal's low percentile is non-positive / near zero).
    """
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = X[i] - asls_baseline(X[i], lam=lam, p=p, niter=niter, method=method)

    floor = np.percentile(out, 5, axis=1)
    assert np.nanmedian(floor) <= np.nanmedian(np.abs(X)) + 1e-6, \
        "baseline-corrected floor is not near zero"
    return out


# ---------------------------------------------------------------------------
# Smoothing / derivative (Savitzky-Golay)
# ---------------------------------------------------------------------------

def smooth(X: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """Per-spectrum Savitzky-Golay smoothing."""
    X = np.asarray(X, dtype=float)
    window = _odd_window(window, X.shape[1], polyorder)
    return savgol_filter(X, window, polyorder, axis=1)


def derivative(
    X: np.ndarray, order: int = 1, window: int = 11, polyorder: int = 3
) -> np.ndarray:
    """Per-spectrum Savitzky-Golay derivative (1st or 2nd order)."""
    X = np.asarray(X, dtype=float)
    window = _odd_window(window, X.shape[1], polyorder)
    return savgol_filter(X, window, polyorder, deriv=order, axis=1)


def _odd_window(window: int, n: int, polyorder: int) -> int:
    window = min(window, n if n % 2 == 1 else n - 1)
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 1 + (polyorder % 2 == 0)
    return window


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def region_select(
    X: np.ndarray, wavenumbers: np.ndarray, lo: float, hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only columns whose wavenumber falls in [lo, hi]."""
    wn = np.asarray(wavenumbers, dtype=float)
    mask = (wn >= lo) & (wn <= hi)
    return X[:, mask], wn[mask]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Toggle set for the preprocessing pipeline. Defaults = raw (control)."""
    smooth: bool = False
    smooth_window: int = 11
    smooth_polyorder: int = 3

    baseline: Optional[str] = None          # None | "asls" | "arpls"
    baseline_lam: float = 1e5
    baseline_p: float = 0.01
    baseline_niter: int = 10

    derivative: int = 0                     # 0 (off) | 1 | 2
    derivative_window: int = 11
    derivative_polyorder: int = 3

    normalize: str = "none"                 # none | minmax | snv | l2

    region: Optional[Tuple[float, float]] = None   # (lo, hi) in cm-1


def preprocess(
    X: np.ndarray,
    config: Optional[PreprocessConfig] = None,
    wavenumbers: Optional[np.ndarray] = None,
):
    """Run the fixed-order, leakage-free pipeline driven by `config`.

    Returns Xp, or (Xp, wn) when a region is selected and `wavenumbers` given.
    """
    if config is None:
        config = PreprocessConfig()

    Xp = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)

    if config.smooth:
        Xp = smooth(Xp, config.smooth_window, config.smooth_polyorder)

    if config.baseline:
        Xp = baseline_correct(
            Xp, method=config.baseline, lam=config.baseline_lam,
            p=config.baseline_p, niter=config.baseline_niter,
        )

    if config.derivative:
        Xp = derivative(
            Xp, order=config.derivative, window=config.derivative_window,
            polyorder=config.derivative_polyorder,
        )

    Xp = normalize(Xp, config.normalize)

    wn_out = None if wavenumbers is None else np.asarray(wavenumbers, dtype=float)
    if config.region is not None:
        if wavenumbers is None:
            raise ValueError("region selection requires `wavenumbers`")
        Xp, wn_out = region_select(Xp, wn_out, *config.region)

    return (Xp, wn_out) if wavenumbers is not None else Xp


if __name__ == "__main__":
    # Tiny smoke test on synthetic spectra (peak + sloping baseline + noise).
    rng = np.random.RandomState(0)
    wn = np.linspace(400, 4000, 600)
    peak = np.exp(-((wn - 1700) ** 2) / (2 * 40 ** 2))
    base = 0.5 + 0.0003 * (wn - 400)
    Xtoy = np.stack([peak * (1 + 0.3 * i) + base + 0.01 * rng.randn(600)
                     for i in range(5)])

    print("raw          ->", preprocess(Xtoy).shape)
    print("snv          -> mean~0 std~1:",
          np.allclose(snv(Xtoy).mean(1), 0, atol=1e-6),
          np.allclose(snv(Xtoy).std(1), 1, atol=1e-6))
    corr = baseline_correct(Xtoy, method="asls")
    print("asls floor   ->", round(float(np.median(np.percentile(corr, 5, axis=1))), 4))
    cfg = PreprocessConfig(baseline="asls", normalize="snv")
    print("asls+snv     ->", preprocess(Xtoy, cfg).shape,
          "(pybaselines)" if _HAS_PYBASELINES else "(scipy fallback)")

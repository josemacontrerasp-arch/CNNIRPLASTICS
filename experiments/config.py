"""Central configuration: paths, class scopes, and the preprocessing grid.

Keeping every path and class-scope decision in one place means the stage scripts
stay short and there is a single source of truth for "what are the six commodity
classes", "where do externals live", "where do outputs go".
"""
from __future__ import annotations

from pathlib import Path

from preprocess import PreprocessConfig

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
FLOPPE_DIR = EXTERNAL_DIR / "flopp_e"      # weathered FTIR (FLOPP-e)
BLOP_DIR = EXTERNAL_DIR / "blop"           # bioplastics FTIR (BLoP)
LAB_FOLDER = DATA_DIR / "ftir_real_world_samples"

# All experiment artifacts land here (organized by stage), kept separate from the
# pre-existing results/ and output/ so nothing old is overwritten.
EXP_OUT = ROOT / "experiments_output"
CACHE_DIR = EXP_OUT / "cache"              # cached aligned .npy matrices
FIG_DIR = EXP_OUT / "figures"
TABLE_DIR = EXP_OUT / "tables"
MODEL_DIR = EXP_OUT / "models"
RESULT_DIR = EXP_OUT / "results"           # machine-readable json/csv per stage

for _d in (EXP_OUT, CACHE_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Class scopes -----------------------------------------------------------
# The TARGET task is the six commodity plastics. PLA/PHA/PBS/PBAT spectra that
# now appear in the regenerated OpenSpecy are NOT target classes; depending on
# the experiment they are either dropped (treated as OOD test material) or
# folded into a single synthetic OTHER class (the "reject option as a class"
# experiment).
SIX = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]
SIX_TO_INT = {c: i for i, c in enumerate(SIX)}
INT_TO_SIX = {i: c for c, i in SIX_TO_INT.items()}

# Bioplastic / non-commodity labels that may appear in OpenSpecy. Anything here
# is non-target: filtered out of the six-class task, or pooled into OTHER.
BIOPLASTIC_LABELS = {"PLA", "PHA", "PBS", "PBAT", "PCL", "PGA", "PEF"}

# Optional "six + OTHER" scope used by the reject-as-a-class experiment.
SIX_PLUS_OTHER = SIX + ["OTHER"]
SIX_PLUS_OTHER_TO_INT = {c: i for i, c in enumerate(SIX_PLUS_OTHER)}

# FLOPP-e/BLoP scoring helpers.
PE_ACCEPT = {"HDPE", "LDPE"}               # FLOPP-e "PE" -> HDPE or LDPE both ok
IN_DIST_MATERIALS = {"PE", "PP", "PS", "PVC", "PET"}

SEED = 0
N_FOLDS = 5


# --- Preprocessing search grid ---------------------------------------------
# A FULL grid (smooth x baseline x derivative x normalize x region) is hundreds
# of fits. We sweep it cheaply with Random Forest (stage 2), then hand only the
# top few configs to the CNN (stage 3). This dict is the menu the RF sweep walks.
#
# Rationale for the chosen axis values:
#   normalize: snv and l2 are the standard FTIR scatter-correctors; minmax is the
#              original CNN's scaling; none is the control.
#   baseline : asls/arpls remove sloping baselines common in ATR-FTIR; None is control.
#   derivative: 1st/2nd Savitzky-Golay derivatives are classic for resolving
#              overlapping bands and killing baseline offset; 0 is control.
#   smooth   : SG smoothing mainly matters before differentiation (noise blow-up).
def sweep_configs() -> "list[tuple[str, PreprocessConfig]]":
    """Return a curated (name, PreprocessConfig) list for the RF sweep.

    Deliberately NOT the full Cartesian product. We vary one family at a time
    around a sensible centre so the sweep stays interpretable and cheap, while
    still covering the questions: does baseline help? does SNV beat min-max?
    do derivatives help? does region-trimming the fingerprint help?
    """
    cfgs: list[tuple[str, PreprocessConfig]] = []

    # 1) Normalization family (no baseline, no derivative) -- isolates scaling.
    for norm in ("none", "minmax", "snv", "l2"):
        cfgs.append((f"norm-{norm}", PreprocessConfig(normalize=norm)))

    # 2) Baseline family (with SNV, the strongest scaler from FTIR practice).
    cfgs.append(("asls+snv", PreprocessConfig(baseline="asls", normalize="snv")))
    cfgs.append(("arpls+snv", PreprocessConfig(baseline="arpls", normalize="snv")))

    # 3) Smoothing + baseline (does SG smoothing before baseline help?).
    cfgs.append(("smooth+asls+snv",
                 PreprocessConfig(smooth=True, baseline="asls", normalize="snv")))

    # 4) Derivative family (SG smoothing on, derivative removes baseline offset).
    cfgs.append(("smooth+d1+snv",
                 PreprocessConfig(smooth=True, derivative=1, normalize="snv")))
    cfgs.append(("smooth+d2+snv",
                 PreprocessConfig(smooth=True, derivative=2, normalize="snv")))

    # 5) Fingerprint-region only (most discriminative bands ~400-1800 cm^-1).
    cfgs.append(("snv-fingerprint",
                 PreprocessConfig(normalize="snv", region=(400.0, 1800.0))))

    return cfgs

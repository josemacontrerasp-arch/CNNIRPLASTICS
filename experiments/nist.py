"""NIST WebBook JCAMP-DX probe spectra for the OOD stress test.

Adapted from data/parsejcamp.py (Jesse's friend's script), but:
  * pure urllib + numpy (no TensorFlow import), so RF-only paths stay light;
  * the synthetic chlorinated-paraffin / PVC probe is DROPPED on request
    (it was manually digitised, not a real measured spectrum);
  * returns raw aligned vectors so the caller applies the same PreprocessConfig
    as training (consistent with experiments/external.py).

These molecules are chosen because they share key bond signatures with specific
plastic classes -- they are deliberate "near-miss" adversarial inputs:

  Stearic acid      -> HDPE/LDPE  (long CH2 chain; C-H stretches)
  Ethylbenzene      -> PS         (aromatic ring + alkyl C-H)
  Ethyl acetate     -> PET        (ester C=O ~1735 + C-O-C)
  Polyisobutylene   -> PP         (polyolefin backbone)  [a.k.a. Vistanex]

Vistanex is a commercial brand of polyisobutylene (PIB), so the PIB probe is
exactly the "Vistanex" material in the brief.

Internet is required to fetch from NIST (works on the user machine; not in the
sandbox). If a fetch fails the probe is returned with an 'error' field so the
stage can report it rather than crash.
"""
from __future__ import annotations

import re
import urllib.request

import numpy as np

PROBE_MOLECULES = [
    dict(name="Stearic acid", proxy_for="HDPE/LDPE",
         reason="Long CH2 chain; shares C-H stretches with polyethylene",
         jcamp_url="https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C57114&Index=0&Type=IR"),
    dict(name="Ethylbenzene", proxy_for="PS",
         reason="Aromatic ring + alkyl C-H; the PS fingerprint",
         jcamp_url="https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C100414&Index=0&Type=IR"),
    dict(name="Ethyl acetate", proxy_for="PET",
         reason="Ester C=O (~1735) and C-O-C; PET's dominant peaks",
         jcamp_url="https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C141786&Index=0&Type=IR"),
    dict(name="Polyisobutylene (Vistanex)", proxy_for="PP",
         reason="Polyolefin backbone; C-H bend/stretch similar to PP",
         jcamp_url="https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C9003274&Index=0&Type=IR"),
]


def parse_jcamp(text: str):
    """Minimal JCAMP-DX IR parser -> (wavenumbers, absorbances), ascending."""
    lines = text.splitlines()
    firstx = lastx = xfactor = yfactor = deltax = None
    npoints = None
    data_start = None
    for i, line in enumerate(lines):
        u = line.upper().strip()
        if u.startswith("##FIRSTX="):
            firstx = float(line.split("=", 1)[1])
        elif u.startswith("##LASTX="):
            lastx = float(line.split("=", 1)[1])
        elif u.startswith("##XFACTOR="):
            xfactor = float(line.split("=", 1)[1])
        elif u.startswith("##YFACTOR="):
            yfactor = float(line.split("=", 1)[1])
        elif u.startswith("##DELTAX="):
            deltax = float(line.split("=", 1)[1])
        elif u.startswith("##NPOINTS="):
            npoints = int(line.split("=", 1)[1])
        elif "##XYDATA" in u or "##XYPOINTS" in u:
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("no XYDATA/XYPOINTS block")
    xfactor = xfactor or 1.0
    yfactor = yfactor or 1.0

    data_lines = []
    for line in lines[data_start:]:
        if line.strip().upper().startswith("##END"):
            break
        data_lines.append(line.strip())

    all_x, all_y = [], []
    for line in data_lines:
        if not line:
            continue
        tokens = re.split(r"[\s,]+", line.strip())
        try:
            nums = [float(t) for t in tokens if t]
        except ValueError:
            continue
        if len(nums) < 2:
            continue
        x_start = nums[0] * xfactor
        y_vals = [v * yfactor for v in nums[1:]]
        if deltax is not None:
            dx = deltax * xfactor
        elif npoints and firstx is not None and lastx is not None:
            dx = (lastx - firstx) / (npoints - 1)
        else:
            dx = None
        if dx is not None:
            for j, yv in enumerate(y_vals):
                all_x.append(x_start + j * dx)
                all_y.append(yv)
        else:
            all_x.append(nums[0] * xfactor)
            all_y.append(nums[1] * yfactor)

    wn = np.asarray(all_x, float)
    ab = np.asarray(all_y, float)
    order = np.argsort(wn)
    return wn[order], ab[order]


def fetch_jcamp(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return parse_jcamp(text)


def load_nist_probes(grid):
    """Fetch + align each probe onto `grid` (clamped). Returns list of row dicts.

    Each row: name, proxy_for, reason, x (raw aligned) or error. All are OOD
    (none of the six commodity plastics), so any high-confidence prediction is a
    false positive driven by shared bond signatures.
    """
    rows = []
    for mol in PROBE_MOLECULES:
        rec = dict(dataset="NIST", file=mol["name"], material=mol["name"],
                   proxy_for=mol["proxy_for"], reason=mol["reason"],
                   truth=None, ood=True)
        try:
            wn, ab = fetch_jcamp(mol["jcamp_url"])
            order = np.argsort(wn)
            res = np.interp(grid, wn[order], ab[order])
            if res.max() == res.min():
                raise ValueError("flat")
            rec["x"] = res.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        rows.append(rec)
    return rows

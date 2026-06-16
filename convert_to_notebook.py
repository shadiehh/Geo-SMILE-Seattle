"""Convert seattle GeoSMILE.py to a Kaggle-ready .ipynb notebook."""

import json, re, pathlib

SRC = pathlib.Path(__file__).parent / "seattle GeoSMILE.py"
DST = pathlib.Path(__file__).parent / "seattle_GeoSMILE.ipynb"

code = SRC.read_text(encoding="utf-8")

# ── Split on step headers ────────────────────────────────────────────────────
HEADER = re.compile(r"(?=# =+\n# (?:Step \d+|Geo-SMILE))")
raw_cells = [c.strip() for c in HEADER.split(code) if c.strip()]

def make_code_cell(src):
    lines = src.splitlines(keepends=True)
    # ensure last line has no trailing newline in the list
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": lines,
    }

def make_md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [src],
    }

cells = []

# Title cell
cells.append(make_md_cell(
    "# Geo-SMILE: Geo + Feature + Cell Explainability Pipeline\n"
    "\n"
    "**Extension of SMILE (Aslansefat et al.) to spatial cases**\n"
    "\n"
    "**How to run:** Import this notebook into Kaggle and attach the dataset `shadiemohammadi/ziqi-seattle`.\n"
    "\n"
    "Branches:\n"
    "- **Geo Branch** — spatial-group perturbations → distributional geo importance (collective)\n"
    "- **Feature Branch** — feature masking with combined `geo_dist` feature → per-point local geo importance + per-point feature importance\n"
    "- **Cell Explainability** — combined (n_points × n_features) importance matrix\n"
    "\n"
    "`geo_dist` = standardised Euclidean distance from dataset centroid, replacing separate UTM_X/UTM_Y in the model (consistent with Chicago dataset design).\n"
    "\n"
    "Metrics: Fidelity · Stability · Sparsity · Entropy (no ground truth required)"
))

# pip install cell
cells.append(make_code_cell("!pip install flaml -q"))

# code cells from source
for block in raw_cells:
    cells.append(make_code_cell(block))

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

DST.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {DST}  ({DST.stat().st_size // 1024} KB, {len(cells)} cells)")

# DA4 Assignment 2 — Economic Growth and CO₂ Emissions

**Author:** Uladzislau Ramaniuk · Central European University

---

## ⚠️ Reproducibility — read before running

> **You must clone the full repository, including the `data/` folder, to reproduce all results.**

The script reads pre-downloaded World Bank data from three CSV files committed to this repo:

| File | Contents |
|---|---|
| `data/wdi_raw.csv` | WDI bulk export — GDP, CO₂, energy use (1960–2024) |
| `data/services_cache.csv` | Services share of GDP via `wbdata` API |
| `data/income_cache.csv` | World Bank income-group classifications |

If these files are absent, the script falls back to a live World Bank API download. That fallback **may fail** (rate limits, API changes) and **will not necessarily reproduce the exact same country sample or numbers cited in the report**.

---

## Quick start

```bash
# 1. Clone – make sure data/ comes with it
git clone https://github.com/Ulad-Ramaniuk/da4-assignment2
cd da4-assignment2

# 2. Install dependencies (requires uv – https://github.com/astral-sh/uv)
uv sync

# 3. Run – a single script produces all results and figures
python analysis.py
```

Outputs are written to `output/figures/`. All quoted numbers are printed to stdout.

---

## Dependencies

Managed by `uv` via `pyproject.toml`. Key packages:

- `pandas`, `numpy`
- `statsmodels`
- `linearmodels` (panel OLS / fixed effects)
- `wbdata` (World Bank API client, used only for the live fallback path)
- `matplotlib`

---

## Repository layout

```
da4-assignment2/
├── analysis.py
├── data/
│   ├── wdi_raw.csv              # ← required for reproducibility
│   ├── services_cache.csv       # ← required for reproducibility
│   └── income_cache.csv         # ← required for reproducibility
├── output/
│   └── figures/                 # generated on first run
├── DA4_Assignment_2_Uladzislau_Ramaniuk.tex
├── DA4_Assignment_2_Uladzislau_Ramaniuk.pdf
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

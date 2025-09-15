# Trading Research Workspace

This workspace holds reusable helpers and packaged study runs. Notebooks produce artifacts into `studies/` so the root stays clean.

## Setup

Mac/Linux:

```bash
cd "Trading Research"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd "Trading Research"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the notebook
Open `notebooks/Intraday_Event_Study.ipynb` and run cells top-to-bottom. The first cell will ensure the venv and dependencies are installed and confirm imports. Data is fetched using Polygon via `helpers.py`.

## Packaged studies
Each run is packaged to `studies/<timestamp>_<slug>/` containing:
- `config.json`: parameters used in the run
- `eda_summary.csv`: key EDA probability tables
- plots: e.g., `hurst_regimes.png`, others
- `model.pkl` (if modeling was performed)
- `README_run.md` with a short summary

## Reusing helpers
`helpers.py` provides:
- Polygon auth and aggregates fetch with retries/pagination
- Feature utilities (VWAP, sigma, ATR, time-of-day)
- Hurst estimation and regime labeling
- Generic event trigger/outcome evaluators
- EDA tables and reliability curves
- Simple modeling helpers and packaging

Import in notebooks:
```python
from helpers import fetch_aggregates, rolling_hurst, package_study
```

## Notes
- Keep `requirements.txt` minimal; the notebook installs missing packages if needed.
- API key: put your Polygon key (single line) in `polygon_key.txt`.

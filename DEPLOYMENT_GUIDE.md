# SIOS — Deployment & Run Guide

Two ways to run the demand-forecasting app. After it starts, open **http://localhost:8501**.

---

## Option A — Docker (recommended, no Python setup)
**Requires:** Docker Desktop installed and running.

```bash
# from the repo root
docker compose up --build
# …or without compose:
docker build -t sios-demand-forecasting .
docker run --rm -p 8501:8501 sios-demand-forecasting
```
Open **http://localhost:8501**. Stop with `Ctrl+C`.

The image builds from `Dockerfile` (Python 3.12 + `requirements-docker.txt`) and bundles `app.py` and the dataset CSVs.

---

## Option B — Python (3.12)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Open **http://localhost:8501**.

---

## Using the app
- **Sidebar** — upload your own demand CSV (columns: `Period` YYYY-MM, `Item`, `Site`, `Demand`; optional `Category`) or use the bundled 20k sample. Toggle **Use Holidays & Events** to see the accuracy difference live.
- **🔮 Forecast Explorer** — actual vs GBDT forecast (+ uncertainty band) per item/warehouse; download all forecasts.
- **📦 Inventory & Reorder** — safety stock & reorder points with lead-time / service-level **what-if sliders**.
- **📊 Model Performance** — baselines vs GBDT (MAE/RMSE/WAPE), and WAPE by demand archetype.
- **🧠 Feature Importance** — what drives the forecast (holiday features highlighted).
- **📋 Data** — dataset preview and summaries.

## Re-generate data / retrain (optional)
```bash
python generate_data.py   # rebuild dataset + holidays calendar (reproducible, fixed seed)
python train_model.py     # retrain GBDT + write metrics & charts to results/
python dist_charts.py     # distribution / EDA charts
python gen_explain.py     # data-generation explanation charts
```

## Troubleshooting
| Problem | Fix |
|---|---|
| Docker not running | Start Docker Desktop, wait for "Engine running", retry. |
| Port 8501 in use | `docker run --rm -p 8600:8501 sios-demand-forecasting` → open http://localhost:8600 |
| `pip install` errors | Use **Python 3.12**, or use the Docker route. |
| Browser doesn't open | Open http://localhost:8501 manually. |

The GBDT uses scikit-learn's `HistGradientBoostingRegressor` (same family as LightGBM/XGBoost) — no extra build tools needed.

# 🚗 Traffic Travel Time Predictor

A single-file Flask web app that predicts **travel time** using Ridge and Lasso regression models trained on simulated traffic data.

---

## Features

- Simulated traffic dataset (500 rows) — no external data files needed
- Preprocessing: label encoding for weather, standard scaling for numerics
- Two models: **Ridge** and **Lasso** regression
- Clean web UI in purple / dark-pink / wine / cream palette
- Displays MAE, RMSE, and R² metrics for both models

---

## Project Structure

```
MFV/
├── traffic_predictor.py   # entire app — Flask routes, models, HTML template
├── requirements.txt       # pip dependencies
└── README.md
```

---

## Run Locally

```bash
pip install -r requirements.txt
python traffic_predictor.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. Set the following:

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python traffic_predictor.py` |
| **Environment** | `Python 3` |

4. Click **Deploy** — Render auto-injects the `PORT` env variable, which the app already reads.

> **Note:** Render's free tier spins down after inactivity; the first request may take ~30 s to wake up.

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `scikit-learn` | Ridge, Lasso, preprocessing, metrics |
| `pandas` | DataFrame handling |
| `numpy` | Numerical operations |

---

## Customisation Tips

- **Change dataset size:** edit `N = 500` near the top of `traffic_predictor.py`
- **Tune regularisation:** adjust `alpha` in `Ridge(alpha=...)` / `Lasso(alpha=...)`
- **Add more features:** extend the `FEATURE_COLS` list and the HTML form together

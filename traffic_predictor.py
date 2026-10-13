"""
Traffic Travel Time Predictor
==============================
Single-file Flask app with Ridge / Lasso regression + a Simulation Section
that shows live Chart.js visualisations of the full ML pipeline.

Run:
  python traffic_predictor.py
  Open http://127.0.0.1:5000
"""

import os, json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── 1. Simulate dataset ───────────────────────────────────────────────────────
np.random.seed(42)
N = 500

WEATHER_OPTIONS = ["Clear", "Rainy", "Foggy", "Snowy"]

weather_raw    = np.random.choice(WEATHER_OPTIONS, N)
traffic_volume = np.random.randint(100, 2000, N)
avg_speed      = np.random.uniform(10, 120, N)
distance       = np.random.uniform(1, 50, N)
time_of_day    = np.random.uniform(0, 24, N)

weather_penalty = np.where(weather_raw == "Clear", 0,
                  np.where(weather_raw == "Rainy", 5,
                  np.where(weather_raw == "Foggy", 8, 15)))

travel_time = (distance / avg_speed) * 60 \
            + 0.005 * traffic_volume \
            + weather_penalty \
            + np.random.normal(0, 2, N)

df = pd.DataFrame({
    "traffic_volume": traffic_volume,
    "avg_speed":      avg_speed,
    "distance":       distance,
    "time_of_day":    time_of_day,
    "weather":        weather_raw,
    "travel_time":    travel_time,
})

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
le = LabelEncoder()
df["weather_enc"] = le.fit_transform(df["weather"])

FEATURE_COLS = ["traffic_volume", "avg_speed", "distance", "time_of_day", "weather_enc"]
TARGET_COL   = "travel_time"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# ── 3. Train models ───────────────────────────────────────────────────────────
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_train, y_train)

def compute_metrics(model, X_t, y_t):
    preds = model.predict(X_t)
    mae   = mean_absolute_error(y_t, preds)
    rmse  = np.sqrt(mean_squared_error(y_t, preds))
    r2    = r2_score(y_t, preds)
    return round(mae, 3), round(rmse, 3), round(r2, 3)

ridge_metrics = compute_metrics(ridge, X_test, y_test)
lasso_metrics = compute_metrics(lasso, X_test, y_test)

# ── Pre-compute simulation data for JS ────────────────────────────────────────
def build_sim_data():
    # Actual vs Predicted (test set) – sample 80 pts for clarity
    idx = np.random.choice(len(y_test), 80, replace=False)
    r_pred = ridge.predict(X_test)
    l_pred = lasso.predict(X_test)

    # Residuals
    r_resid = (y_test - r_pred).tolist()
    l_resid = (y_test - l_pred).tolist()

    # Feature importance (|coefficient|)
    feat_labels = ["Traffic Vol", "Avg Speed", "Distance", "Time of Day", "Weather"]
    ridge_coef  = np.abs(ridge.coef_).tolist()
    lasso_coef  = np.abs(lasso.coef_).tolist()

    # Weather distribution
    weather_counts = df["weather"].value_counts()
    weather_labels = weather_counts.index.tolist()
    weather_vals   = weather_counts.values.tolist()

    # Travel time histogram (30 bins)
    hist_vals, hist_edges = np.histogram(y, bins=30)
    hist_labels = [f"{e:.1f}" for e in hist_edges[:-1]]

    # Speed vs Travel Time scatter (sample 120)
    s_idx = np.random.choice(N, 120, replace=False)
    speed_scatter  = df["avg_speed"].values[s_idx].tolist()
    ttime_scatter  = df["travel_time"].values[s_idx].tolist()

    return {
        "actual":          y_test[idx].tolist(),
        "ridge_pred":      r_pred[idx].tolist(),
        "lasso_pred":      l_pred[idx].tolist(),
        "r_resid":         r_resid,
        "l_resid":         l_resid,
        "feat_labels":     feat_labels,
        "ridge_coef":      ridge_coef,
        "lasso_coef":      lasso_coef,
        "weather_labels":  weather_labels,
        "weather_vals":    weather_vals,
        "hist_labels":     hist_labels,
        "hist_vals":       hist_vals.tolist(),
        "speed_scatter":   speed_scatter,
        "ttime_scatter":   ttime_scatter,
        # Dataset stats
        "n_samples":       N,
        "n_train":         len(y_train),
        "n_test":          len(y_test),
        "n_features":      len(FEATURE_COLS),
        "ridge_r2":        ridge_metrics[2],
        "lasso_r2":        lasso_metrics[2],
        "ridge_mae":       ridge_metrics[0],
        "lasso_mae":       lasso_metrics[0],
        "ridge_rmse":      ridge_metrics[1],
        "lasso_rmse":      lasso_metrics[1],
    }

SIM_DATA = build_sim_data()

# ── 4. HTML template ──────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Traffic Travel Time Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet" />
  <style>
    :root {
      --purple:   #7c3aed;
      --dpink:    #db2777;
      --wine:     #9f1239;
      --cream:    #0f0a1e;
      --glass:    rgba(255,255,255,0.04);
      --glassbrd: rgba(255,255,255,0.12);
      --neon1:    #a855f7;
      --neon2:    #ec4899;
      --neon3:    #06b6d4;
      --text:     #f1e8ff;
      --muted:    #9d8bb0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Outfit', sans-serif;
      background: var(--cream);
      color: var(--text);
      min-height: 100vh;
      padding: 24px 16px 60px;
      background-image:
        radial-gradient(ellipse at 20% 10%, rgba(124,58,237,.25) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(219,39,119,.2) 0%, transparent 50%);
    }

    /* ── HEADER ── */
    header {
      text-align: center;
      padding: 40px 20px 32px;
      position: relative;
    }
    header h1 {
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 800;
      background: linear-gradient(90deg, var(--neon1), var(--neon2), var(--neon3));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -1px;
    }
    header p { color: var(--muted); margin-top: 8px; font-size: .95rem; }

    /* ── NAV TABS ── */
    .tabs {
      display: flex;
      justify-content: center;
      gap: 10px;
      margin: 28px auto 36px;
      max-width: 500px;
    }
    .tab-btn {
      flex: 1;
      padding: 12px 20px;
      border: 1px solid var(--glassbrd);
      border-radius: 50px;
      background: var(--glass);
      color: var(--muted);
      font-family: 'Outfit', sans-serif;
      font-size: .95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all .25s;
      backdrop-filter: blur(8px);
    }
    .tab-btn:hover { color: var(--text); border-color: var(--neon1); }
    .tab-btn.active {
      background: linear-gradient(135deg, var(--purple), var(--dpink));
      color: #fff;
      border-color: transparent;
      box-shadow: 0 0 20px rgba(168,85,247,.5);
    }

    /* ── SECTIONS ── */
    .section { display: none; }
    .section.active { display: block; }

    /* ── GLASS CARD ── */
    .card {
      background: var(--glass);
      border: 1px solid var(--glassbrd);
      border-radius: 20px;
      padding: 30px;
      max-width: 700px;
      margin: 0 auto 28px auto;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,.4);
    }
    .card h2 {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--neon1);
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card h2 .pill {
      font-size: .7rem;
      background: rgba(168,85,247,.2);
      border: 1px solid var(--neon1);
      color: var(--neon1);
      padding: 2px 10px;
      border-radius: 20px;
      font-weight: 600;
    }

    /* ── FORM ── */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    @media(max-width:520px){ .grid{ grid-template-columns: 1fr; } }

    label {
      display: block;
      font-size: .78rem;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: .5px;
    }
    input, select {
      width: 100%;
      padding: 10px 14px;
      background: rgba(255,255,255,.07);
      border: 1px solid var(--glassbrd);
      border-radius: 10px;
      font-size: .95rem;
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      outline: none;
      transition: border .2s, box-shadow .2s;
    }
    input:focus, select:focus {
      border-color: var(--neon1);
      box-shadow: 0 0 0 3px rgba(168,85,247,.2);
    }
    select option { background: #1a0a2e; color: var(--text); }

    .btn-row { display:flex; gap:14px; margin-top:24px; flex-wrap:wrap; }
    button[type=submit] {
      flex: 1;
      padding: 13px 0;
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      font-family: 'Outfit', sans-serif;
      letter-spacing: .4px;
      transition: transform .15s, box-shadow .15s;
      position: relative;
      overflow: hidden;
    }
    button[type=submit]:hover { transform: translateY(-3px); }
    .btn-ridge {
      background: linear-gradient(135deg, #7c3aed, #4f46e5);
      color: #fff;
      box-shadow: 0 4px 20px rgba(124,58,237,.4);
    }
    .btn-ridge:hover { box-shadow: 0 8px 30px rgba(124,58,237,.6); }
    .btn-lasso {
      background: linear-gradient(135deg, #db2777, #9f1239);
      color: #fff;
      box-shadow: 0 4px 20px rgba(219,39,119,.4);
    }
    .btn-lasso:hover { box-shadow: 0 8px 30px rgba(219,39,119,.6); }

    /* ── RESULT BOX ── */
    .result-box {
      background: linear-gradient(135deg, rgba(124,58,237,.15), rgba(219,39,119,.15));
      border: 1px solid rgba(168,85,247,.3);
      border-radius: 16px;
      padding: 22px 26px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }
    .result-box::before {
      content: '';
      position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(168,85,247,.05), rgba(236,72,153,.05));
      pointer-events: none;
    }
    .result-box .label { font-size: .75rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .8px; }
    .result-box .value {
      font-size: 3rem;
      font-weight: 800;
      background: linear-gradient(90deg, var(--neon1), var(--neon2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      line-height: 1.1;
      margin: 4px 0 8px;
    }
    .model-tag {
      display: inline-block;
      background: rgba(168,85,247,.2);
      border: 1px solid var(--neon1);
      color: var(--neon1);
      font-size: .75rem;
      font-weight: 700;
      padding: 3px 14px;
      border-radius: 20px;
    }

    /* ── METRICS TABLE ── */
    table { width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: 12px; }
    th {
      background: rgba(124,58,237,.3);
      color: var(--neon1);
      padding: 10px 14px;
      text-align: left;
      font-weight: 700;
      font-size: .8rem;
      text-transform: uppercase;
      letter-spacing: .5px;
    }
    td { padding: 10px 14px; border-bottom: 1px solid var(--glassbrd); color: var(--text); }
    tr:nth-child(even) td { background: rgba(255,255,255,.03); }
    .active-row td { color: var(--neon2); font-weight: 700; }

    /* ════════════════ SIMULATION SECTION ════════════════ */

    .sim-header {
      text-align: center;
      margin-bottom: 36px;
    }
    .sim-header h2 {
      font-size: clamp(1.4rem, 3vw, 2rem);
      font-weight: 800;
      background: linear-gradient(90deg, var(--neon3), var(--neon1), var(--neon2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    .sim-header p { color: var(--muted); font-size: .9rem; }

    /* Pipeline steps */
    .pipeline {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 0;
      margin: 0 auto 40px;
      max-width: 860px;
    }
    .pipe-step {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      animation: popIn .5s ease both;
    }
    .pipe-step:nth-child(1)  { animation-delay: .1s; }
    .pipe-step:nth-child(3)  { animation-delay: .2s; }
    .pipe-step:nth-child(5)  { animation-delay: .3s; }
    .pipe-step:nth-child(7)  { animation-delay: .4s; }
    .pipe-step:nth-child(9)  { animation-delay: .5s; }
    @keyframes popIn {
      from { opacity:0; transform: scale(.6); }
      to   { opacity:1; transform: scale(1); }
    }
    .pipe-icon {
      width: 64px; height: 64px;
      border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.6rem;
      position: relative;
    }
    .pipe-icon::after {
      content: '';
      position: absolute; inset: -2px;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--neon1), var(--neon2));
      z-index: -1;
      opacity: .7;
    }
    .pipe-icon.c1 { background: rgba(124,58,237,.3); }
    .pipe-icon.c2 { background: rgba(6,182,212,.3); }
    .pipe-icon.c3 { background: rgba(219,39,119,.3); }
    .pipe-icon.c4 { background: rgba(168,85,247,.3); }
    .pipe-icon.c5 { background: rgba(16,185,129,.3); }
    .pipe-label { font-size: .72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; text-align: center; }
    .pipe-arrow {
      color: var(--glassbrd);
      font-size: 1.4rem;
      padding: 0 6px;
      animation: pulse 1.8s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }

    /* Stat cards */
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px;
      max-width: 960px;
      margin: 0 auto 40px;
    }
    .stat-card {
      background: var(--glass);
      border: 1px solid var(--glassbrd);
      border-radius: 16px;
      padding: 18px 16px;
      text-align: center;
      backdrop-filter: blur(10px);
      position: relative;
      overflow: hidden;
      animation: fadeUp .6s ease both;
    }
    .stat-card:nth-child(1){ animation-delay:.05s; }
    .stat-card:nth-child(2){ animation-delay:.1s; }
    .stat-card:nth-child(3){ animation-delay:.15s; }
    .stat-card:nth-child(4){ animation-delay:.2s; }
    .stat-card:nth-child(5){ animation-delay:.25s; }
    .stat-card:nth-child(6){ animation-delay:.3s; }
    @keyframes fadeUp {
      from { opacity:0; transform: translateY(20px); }
      to   { opacity:1; transform: translateY(0); }
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: -40%; left: -40%;
      width: 80%; height: 80%;
      background: radial-gradient(circle, rgba(168,85,247,.15), transparent 70%);
      pointer-events: none;
    }
    .stat-num {
      font-size: 2rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--neon1), var(--neon3));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .stat-lbl { font-size: .72rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; margin-top: 4px; }

    /* Chart grid */
    .chart-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      max-width: 960px;
      margin: 0 auto 30px;
    }
    @media(max-width: 640px) { .chart-grid { grid-template-columns: 1fr; } }

    .chart-card {
      background: var(--glass);
      border: 1px solid var(--glassbrd);
      border-radius: 20px;
      padding: 22px;
      backdrop-filter: blur(12px);
      animation: fadeUp .7s ease both;
      position: relative;
      overflow: hidden;
    }
    .chart-card::after {
      content: '';
      position: absolute;
      bottom: -30px; right: -30px;
      width: 100px; height: 100px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(168,85,247,.08), transparent 70%);
    }
    .chart-card.wide { grid-column: span 2; }
    @media(max-width:640px){ .chart-card.wide{ grid-column: span 1; } }

    .chart-title {
      font-size: .82rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .8px;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .chart-title .dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--neon1);
    }
    .chart-title .dot.pink { background: var(--neon2); }
    .chart-title .dot.cyan { background: var(--neon3); }

    canvas { width: 100% !important; }

    /* Model badge row */
    .model-compare {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      max-width: 960px;
      margin: 0 auto 30px;
    }
    @media(max-width:540px){ .model-compare{ grid-template-columns:1fr; } }
    .model-card {
      border-radius: 18px;
      padding: 22px;
      backdrop-filter: blur(12px);
      animation: fadeUp .6s .3s ease both;
    }
    .model-card.ridge-card {
      background: linear-gradient(135deg, rgba(124,58,237,.2), rgba(79,70,229,.1));
      border: 1px solid rgba(124,58,237,.4);
    }
    .model-card.lasso-card {
      background: linear-gradient(135deg, rgba(219,39,119,.2), rgba(159,18,57,.1));
      border: 1px solid rgba(219,39,119,.4);
    }
    .model-card .m-name {
      font-size: 1.1rem; font-weight: 800; margin-bottom: 12px;
    }
    .ridge-card .m-name { color: #a78bfa; }
    .lasso-card .m-name { color: #f472b6; }
    .metric-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
    .metric-row:last-child { margin-bottom:0; }
    .metric-row .m-lbl { font-size: .78rem; color: var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
    .metric-row .m-val { font-size: 1.1rem; font-weight: 800; }
    .ridge-card .m-val { color: #c4b5fd; }
    .lasso-card .m-val { color: #f9a8d4; }
    .bar-bg { background: rgba(255,255,255,.07); border-radius: 20px; height: 6px; margin-top:4px; }
    .bar-fill { height: 6px; border-radius: 20px; transition: width 1s ease; }
    .ridge-bar { background: linear-gradient(90deg, #7c3aed, #4f46e5); }
    .lasso-bar { background: linear-gradient(90deg, #db2777, #9f1239); }

    footer {
      text-align: center;
      font-size: .78rem;
      color: var(--muted);
      margin-top: 20px;
    }
  </style>
</head>
<body>

<header>
  <h1>🚗 Traffic Travel Time Predictor</h1>
  <p>Ridge &amp; Lasso Regression &mdash; trained on simulated traffic data</p>
</header>

<!-- NAV TABS -->
<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('predictor', this)" id="tab-predictor">
    🎯 Predictor
  </button>
  <button class="tab-btn" onclick="switchTab('simulation', this)" id="tab-simulation">
    🧬 Simulation
  </button>
</div>

<!-- ═══════════════════ PREDICTOR SECTION ═══════════════════ -->
<div id="predictor" class="section active">

  <div class="card">
    <h2>Enter Traffic Features <span class="pill">LIVE</span></h2>
    <form method="POST" action="/">
      <div class="grid">
        <div>
          <label for="traffic_volume">Traffic Volume (vehicles/hr)</label>
          <input id="traffic_volume" name="traffic_volume" type="number"
                 min="100" max="2000" step="10"
                 value="{{ form.traffic_volume | default(800) }}" required />
        </div>
        <div>
          <label for="avg_speed">Average Speed (km/h)</label>
          <input id="avg_speed" name="avg_speed" type="number"
                 min="10" max="120" step="1"
                 value="{{ form.avg_speed | default(60) }}" required />
        </div>
        <div>
          <label for="distance">Distance (km)</label>
          <input id="distance" name="distance" type="number"
                 min="1" max="50" step="0.5"
                 value="{{ form.distance | default(15) }}" required />
        </div>
        <div>
          <label for="time_of_day">Time of Day (0–24 hr)</label>
          <input id="time_of_day" name="time_of_day" type="number"
                 min="0" max="24" step="0.5"
                 value="{{ form.time_of_day | default(8) }}" required />
        </div>
        <div>
          <label for="weather">Weather Condition</label>
          <select id="weather" name="weather">
            {% for w in weather_options %}
              <option value="{{ w }}" {% if form.weather == w %}selected{% endif %}>{{ w }}</option>
            {% endfor %}
          </select>
        </div>
      </div>

      <div class="btn-row">
        <button type="submit" name="model" value="ridge" class="btn-ridge">
          🔵 Predict with Ridge
        </button>
        <button type="submit" name="model" value="lasso" class="btn-lasso">
          🌸 Predict with Lasso
        </button>
      </div>
    </form>
  </div>

  {% if prediction is not none %}
  <div class="card">
    <h2>Prediction Result</h2>
    <div class="result-box">
      <div class="label">Estimated Travel Time</div>
      <div class="value">{{ prediction }} min</div>
      <span class="model-tag">{{ model_used }} Regression</span>
    </div>

    <h2>Model Performance on Test Set</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>
      </thead>
      <tbody>
        <tr class="{% if model_used == 'Ridge' %}active-row{% endif %}">
          <td>Ridge</td>
          <td>{{ metrics.ridge.mae }}</td>
          <td>{{ metrics.ridge.rmse }}</td>
          <td>{{ metrics.ridge.r2 }}</td>
        </tr>
        <tr class="{% if model_used == 'Lasso' %}active-row{% endif %}">
          <td>Lasso</td>
          <td>{{ metrics.lasso.mae }}</td>
          <td>{{ metrics.lasso.rmse }}</td>
          <td>{{ metrics.lasso.r2 }}</td>
        </tr>
      </tbody>
    </table>
  </div>
  {% endif %}

</div><!-- #predictor -->


<!-- ═══════════════════ SIMULATION SECTION ═══════════════════ -->
<div id="simulation" class="section">

  <div class="sim-header">
    <h2>🔬 Under The Hood</h2>
    <p>Full ML pipeline visualised — from raw data to prediction</p>
  </div>

  <!-- ── Pipeline Steps ── -->
  <div class="pipeline">
    <div class="pipe-step">
      <div class="pipe-icon c1">📊</div>
      <div class="pipe-label">Raw<br>Data</div>
    </div>
    <div class="pipe-arrow">▶</div>
    <div class="pipe-step">
      <div class="pipe-icon c2">🔤</div>
      <div class="pipe-label">Label<br>Encode</div>
    </div>
    <div class="pipe-arrow">▶</div>
    <div class="pipe-step">
      <div class="pipe-icon c3">📐</div>
      <div class="pipe-label">Standard<br>Scale</div>
    </div>
    <div class="pipe-arrow">▶</div>
    <div class="pipe-step">
      <div class="pipe-icon c4">✂️</div>
      <div class="pipe-label">Train/<br>Test Split</div>
    </div>
    <div class="pipe-arrow">▶</div>
    <div class="pipe-step">
      <div class="pipe-icon c5">🤖</div>
      <div class="pipe-label">Ridge &<br>Lasso</div>
    </div>
  </div>

  <!-- ── Animated Stats ── -->
  <div class="stat-grid">
    <div class="stat-card">
      <div class="stat-num" data-target="{{ sim.n_samples }}">0</div>
      <div class="stat-lbl">Total Samples</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="{{ sim.n_train }}">0</div>
      <div class="stat-lbl">Train Set</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="{{ sim.n_test }}">0</div>
      <div class="stat-lbl">Test Set</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="{{ sim.n_features }}">0</div>
      <div class="stat-lbl">Features</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="{{ (sim.ridge_r2 * 100)|round(1) }}" data-suffix="%">0</div>
      <div class="stat-lbl">Ridge R²</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" data-target="{{ (sim.lasso_r2 * 100)|round(1) }}" data-suffix="%">0</div>
      <div class="stat-lbl">Lasso R²</div>
    </div>
  </div>

  <!-- ── Model Comparison ── -->
  <div class="model-compare">
    <div class="model-card ridge-card">
      <div class="m-name">🔵 Ridge Regression</div>
      <div class="metric-row">
        <span class="m-lbl">MAE</span>
        <span class="m-val">{{ sim.ridge_mae }} min</span>
      </div>
      <div class="metric-row">
        <span class="m-lbl">RMSE</span>
        <span class="m-val">{{ sim.ridge_rmse }} min</span>
      </div>
      <div class="metric-row">
        <span class="m-lbl">R² Score</span>
        <span class="m-val">{{ sim.ridge_r2 }}</span>
      </div>
      <div class="bar-bg"><div class="bar-fill ridge-bar" style="width:{{ (sim.ridge_r2*100)|round(0) }}%"></div></div>
    </div>
    <div class="model-card lasso-card">
      <div class="m-name">🌸 Lasso Regression</div>
      <div class="metric-row">
        <span class="m-lbl">MAE</span>
        <span class="m-val">{{ sim.lasso_mae }} min</span>
      </div>
      <div class="metric-row">
        <span class="m-lbl">RMSE</span>
        <span class="m-val">{{ sim.lasso_rmse }} min</span>
      </div>
      <div class="metric-row">
        <span class="m-lbl">R² Score</span>
        <span class="m-val">{{ sim.lasso_r2 }}</span>
      </div>
      <div class="bar-bg"><div class="bar-fill lasso-bar" style="width:{{ (sim.lasso_r2*100)|round(0) }}%"></div></div>
    </div>
  </div>

  <!-- ── Charts ── -->
  <div class="chart-grid">

    <!-- Actual vs Predicted -->
    <div class="chart-card wide">
      <div class="chart-title"><span class="dot"></span> Actual vs Predicted Travel Time (Test Set)</div>
      <canvas id="scatterChart" height="220"></canvas>
    </div>

    <!-- Feature Importance -->
    <div class="chart-card">
      <div class="chart-title"><span class="dot cyan"></span> Feature Importance (|Coefficient|)</div>
      <canvas id="coefChart" height="230"></canvas>
    </div>

    <!-- Residuals Distribution -->
    <div class="chart-card">
      <div class="chart-title"><span class="dot pink"></span> Residuals Distribution</div>
      <canvas id="residChart" height="230"></canvas>
    </div>

    <!-- Travel Time Distribution -->
    <div class="chart-card">
      <div class="chart-title"><span class="dot"></span> Travel Time Distribution</div>
      <canvas id="histChart" height="230"></canvas>
    </div>

    <!-- Weather Breakdown -->
    <div class="chart-card">
      <div class="chart-title"><span class="dot pink"></span> Weather Condition Breakdown</div>
      <canvas id="weatherChart" height="230"></canvas>
    </div>

  </div><!-- .chart-grid -->

</div><!-- #simulation -->

<footer>Traffic Travel Time Predictor &nbsp;|&nbsp; Ridge &amp; Lasso Regression</footer>

<!-- ═══════ SCRIPTS ═══════ -->
<script>
  // ── Tab switcher ───────────────────────────────────────────────────────────
  function switchTab(id, btn) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
    if (id === 'simulation') initSim();
  }

  // ── Animated counters ──────────────────────────────────────────────────────
  function animateCounters() {
    document.querySelectorAll('.stat-num[data-target]').forEach(el => {
      const target = parseFloat(el.dataset.target);
      const suffix = el.dataset.suffix || '';
      const isFloat = target % 1 !== 0;
      let start = 0;
      const dur = 1200;
      const step = 16;
      const inc  = target / (dur / step);
      const timer = setInterval(() => {
        start += inc;
        if (start >= target) { start = target; clearInterval(timer); }
        el.textContent = isFloat ? start.toFixed(1) + suffix : Math.floor(start) + suffix;
      }, step);
    });
  }

  // ── Sim init guard ─────────────────────────────────────────────────────────
  let simReady = false;
  function initSim() {
    if (simReady) return;
    simReady = true;
    animateCounters();
    buildCharts();
  }

  // ── Simulation data from Python ────────────────────────────────────────────
  const SD = {{ sim_json | safe }};

  // ── Chart defaults ─────────────────────────────────────────────────────────
  Chart.defaults.color = '#9d8bb0';
  Chart.defaults.font.family = "'Outfit', sans-serif";

  function buildCharts() {

    const neon1 = '#a855f7', neon2 = '#ec4899', neon3 = '#06b6d4';
    const glass = 'rgba(255,255,255,0.04)';

    const baseOpts = (extra={}) => ({
      responsive: true,
      animation: { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend: { labels: { boxWidth: 12, color: '#9d8bb0', font: { size: 11 } } },
        tooltip: { backgroundColor: 'rgba(15,10,30,.9)', borderColor: 'rgba(168,85,247,.3)', borderWidth: 1 },
      },
      ...extra
    });

    // 1. Actual vs Predicted scatter
    new Chart(document.getElementById('scatterChart'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Ridge',
            data: SD.actual.map((a, i) => ({ x: a, y: SD.ridge_pred[i] })),
            backgroundColor: neon1 + '99',
            pointRadius: 4,
          },
          {
            label: 'Lasso',
            data: SD.actual.map((a, i) => ({ x: a, y: SD.lasso_pred[i] })),
            backgroundColor: neon2 + '99',
            pointRadius: 4,
          },
          {
            label: 'Perfect Fit',
            type: 'line',
            data: [
              { x: Math.min(...SD.actual), y: Math.min(...SD.actual) },
              { x: Math.max(...SD.actual), y: Math.max(...SD.actual) },
            ],
            borderColor: neon3,
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: baseOpts({
        scales: {
          x: { title: { display: true, text: 'Actual (min)', color: '#9d8bb0' }, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#9d8bb0' } },
          y: { title: { display: true, text: 'Predicted (min)', color: '#9d8bb0' }, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#9d8bb0' } },
        }
      })
    });

    // 2. Feature importance (grouped bar)
    new Chart(document.getElementById('coefChart'), {
      type: 'bar',
      data: {
        labels: SD.feat_labels,
        datasets: [
          {
            label: 'Ridge |coef|',
            data: SD.ridge_coef,
            backgroundColor: neon1 + 'cc',
            borderRadius: 6,
          },
          {
            label: 'Lasso |coef|',
            data: SD.lasso_coef,
            backgroundColor: neon2 + 'cc',
            borderRadius: 6,
          }
        ]
      },
      options: baseOpts({
        scales: {
          x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#9d8bb0', font:{size:10} } },
          y: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#9d8bb0' } }
        }
      })
    });

    // 3. Residuals histogram
    function makeHist(arr, bins=20) {
      const mn = Math.min(...arr), mx = Math.max(...arr);
      const w  = (mx - mn) / bins;
      const cnts = Array(bins).fill(0);
      const lbls = [];
      for (let i = 0; i < bins; i++) lbls.push((mn + i * w).toFixed(1));
      arr.forEach(v => {
        let b = Math.floor((v - mn) / w);
        if (b >= bins) b = bins - 1;
        cnts[b]++;
      });
      return { lbls, cnts };
    }
    const rh = makeHist(SD.r_resid);
    const lh = makeHist(SD.l_resid);
    new Chart(document.getElementById('residChart'), {
      type: 'bar',
      data: {
        labels: rh.lbls,
        datasets: [
          { label: 'Ridge Residuals', data: rh.cnts, backgroundColor: neon1 + 'aa', borderRadius: 4 },
          { label: 'Lasso Residuals', data: lh.cnts, backgroundColor: neon2 + 'aa', borderRadius: 4 },
        ]
      },
      options: baseOpts({
        scales: {
          x: { ticks: { color: '#9d8bb0', maxTicksLimit: 8, font:{size:9} }, grid: { color: 'rgba(255,255,255,.05)' }, title:{display:true,text:'Error (min)',color:'#9d8bb0'} },
          y: { ticks: { color: '#9d8bb0' }, grid: { color: 'rgba(255,255,255,.05)' }, title:{display:true,text:'Count',color:'#9d8bb0'} }
        }
      })
    });

    // 4. Travel time histogram
    new Chart(document.getElementById('histChart'), {
      type: 'bar',
      data: {
        labels: SD.hist_labels,
        datasets: [{
          label: 'Travel Time (min)',
          data: SD.hist_vals,
          backgroundColor: `linear-gradient(${neon3}, ${neon1})`,
          backgroundColor: neon3 + 'bb',
          borderRadius: 4,
        }]
      },
      options: baseOpts({
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#9d8bb0', maxTicksLimit: 8, font:{size:9} }, grid: { color: 'rgba(255,255,255,.05)' }, title:{display:true,text:'Travel Time (min)',color:'#9d8bb0'} },
          y: { ticks: { color: '#9d8bb0' }, grid: { color: 'rgba(255,255,255,.05)' }, title:{display:true,text:'Count',color:'#9d8bb0'} }
        }
      })
    });

    // 5. Weather doughnut
    new Chart(document.getElementById('weatherChart'), {
      type: 'doughnut',
      data: {
        labels: SD.weather_labels,
        datasets: [{
          data: SD.weather_vals,
          backgroundColor: [neon1 + 'cc', neon2 + 'cc', neon3 + 'cc', '#f59e0b' + 'cc'],
          hoverOffset: 10,
          borderWidth: 2,
          borderColor: 'rgba(15,10,30,.8)',
        }]
      },
      options: baseOpts({
        cutout: '60%',
        plugins: {
          legend: { position: 'bottom', labels: { color: '#9d8bb0', padding: 14 } }
        }
      })
    });
  }
</script>

</body>
</html>
"""

# ── 5. Flask routes ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_used = None
    form       = {}

    if request.method == "POST":
        try:
            form = request.form.to_dict()
            tv   = float(form["traffic_volume"])
            spd  = float(form["avg_speed"])
            dst  = float(form["distance"])
            tod  = float(form["time_of_day"])
            wthr = form.get("weather", "Clear")
            mdl  = form.get("model", "ridge")

            wthr_enc   = le.transform([wthr])[0]
            raw_vec    = np.array([[tv, spd, dst, tod, wthr_enc]])
            scaled_vec = scaler.transform(raw_vec)

            if mdl == "lasso":
                pred_val   = lasso.predict(scaled_vec)[0]
                model_used = "Lasso"
            else:
                pred_val   = ridge.predict(scaled_vec)[0]
                model_used = "Ridge"

            prediction = round(pred_val, 2)

        except Exception as exc:
            prediction = f"Error: {exc}"

    metrics = {
        "ridge": {"mae": ridge_metrics[0], "rmse": ridge_metrics[1], "r2": ridge_metrics[2]},
        "lasso": {"mae": lasso_metrics[0], "rmse": lasso_metrics[1], "r2": lasso_metrics[2]},
    }

    return render_template_string(
        HTML,
        prediction      = prediction,
        model_used      = model_used,
        form            = form,
        weather_options = WEATHER_OPTIONS,
        metrics         = metrics,
        sim             = SIM_DATA,
        sim_json        = json.dumps(SIM_DATA),
    )


# ── 6. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

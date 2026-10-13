"""
Traffic Travel Time Predictor
==============================
A single-file Flask web app that:
  - Simulates a traffic dataset
  - Trains Ridge and Lasso regression models
  - Accepts user input and predicts travel time
  - Displays MAE, RMSE, and R² metrics

Run locally:
  pip install flask scikit-learn pandas
  python traffic_predictor.py
  Then open http://127.0.0.1:5000

Deploy to Heroku:
  1. Create a Procfile with: web: python traffic_predictor.py
  2. Set PORT env var; Flask will use it automatically.
  3. git init, heroku create, git push heroku main

Deploy to Render / Railway / Fly.io:
  - Point the start command to: python traffic_predictor.py
  - They auto-detect Flask apps via the PORT env variable.
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── 1. Simulate a small traffic dataset ────────────────────────────────────
np.random.seed(42)
N = 500

WEATHER_OPTIONS = ["Clear", "Rainy", "Foggy", "Snowy"]

weather_raw = np.random.choice(WEATHER_OPTIONS, N)
traffic_volume = np.random.randint(100, 2000, N)        # vehicles/hour
avg_speed      = np.random.uniform(10, 120, N)          # km/h
distance       = np.random.uniform(1, 50, N)            # km
time_of_day    = np.random.uniform(0, 24, N)            # hour (0-24)

# Target: travel_time (minutes) — loosely physics-based with noise
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

# ── 2. Preprocessing ───────────────────────────────────────────────────────
# Encode categorical weather column
le = LabelEncoder()
df["weather_enc"] = le.fit_transform(df["weather"])

FEATURE_COLS = ["traffic_volume", "avg_speed", "distance", "time_of_day", "weather_enc"]
TARGET_COL   = "travel_time"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# Scale numeric features for regularised models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 3. Train Ridge and Lasso models ────────────────────────────────────────
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1, max_iter=5000)
lasso.fit(X_train, y_train)

def compute_metrics(model, X_t, y_t):
    """Return MAE, RMSE, R² for a fitted model."""
    preds = model.predict(X_t)
    mae   = mean_absolute_error(y_t, preds)
    rmse  = np.sqrt(mean_squared_error(y_t, preds))
    r2    = r2_score(y_t, preds)
    return round(mae, 3), round(rmse, 3), round(r2, 3)

ridge_metrics = compute_metrics(ridge, X_test, y_test)
lasso_metrics = compute_metrics(lasso, X_test, y_test)

# ── 4. HTML template (inline) ──────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Traffic Travel Time Predictor</title>
  <style>
    /* ── Palette: purple, dark-pink, wine, cream ── */
    :root {
      --purple:    #6a0dad;
      --dpink:     #c2185b;
      --wine:      #722f37;
      --cream:     #fdf6ec;
      --light-pur: #ede0f7;
      --text:      #2e0a3a;
      --border:    #d4b8e0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: Georgia, serif;
      background: var(--cream);
      color: var(--text);
      min-height: 100vh;
      padding: 30px 16px;
    }

    header {
      background: linear-gradient(135deg, var(--wine), var(--purple));
      color: #fff;
      text-align: center;
      padding: 28px 20px;
      border-radius: 12px;
      margin-bottom: 30px;
      box-shadow: 0 4px 18px rgba(106,13,173,.3);
    }
    header h1 { font-size: 1.9rem; letter-spacing: 1px; }
    header p  { margin-top: 6px; font-size: .95rem; opacity: .88; }

    .card {
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 28px;
      max-width: 680px;
      margin: 0 auto 28px auto;
      box-shadow: 0 2px 12px rgba(114,47,55,.1);
    }
    .card h2 {
      color: var(--purple);
      margin-bottom: 18px;
      font-size: 1.2rem;
      border-bottom: 2px solid var(--light-pur);
      padding-bottom: 8px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media(max-width:500px){ .grid{ grid-template-columns:1fr; } }

    label {
      display: block;
      font-size: .85rem;
      font-weight: bold;
      color: var(--wine);
      margin-bottom: 4px;
    }
    input, select {
      width: 100%;
      padding: 9px 12px;
      border: 1px solid var(--border);
      border-radius: 7px;
      font-size: .95rem;
      background: var(--light-pur);
      color: var(--text);
      outline: none;
      transition: border .2s;
    }
    input:focus, select:focus { border-color: var(--purple); }

    .btn-row {
      display: flex;
      gap: 14px;
      margin-top: 22px;
      flex-wrap: wrap;
    }
    button {
      flex: 1;
      padding: 11px 0;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      font-family: Georgia, serif;
      letter-spacing: .5px;
      transition: opacity .18s, transform .12s;
    }
    button:hover { opacity: .88; transform: translateY(-2px); }

    .btn-ridge { background: var(--purple); color: #fff; }
    .btn-lasso { background: var(--dpink);  color: #fff; }

    /* Result box */
    .result-box {
      background: linear-gradient(135deg, var(--light-pur), #ffe6f0);
      border-left: 5px solid var(--wine);
      border-radius: 10px;
      padding: 18px 22px;
      margin-bottom: 22px;
    }
    .result-box .label { font-size: .8rem; color: var(--wine); font-weight: bold; }
    .result-box .value { font-size: 2.2rem; color: var(--purple); font-weight: bold; }
    .result-box .model-tag {
      display: inline-block;
      background: var(--wine);
      color: #fff;
      font-size: .75rem;
      padding: 2px 10px;
      border-radius: 20px;
      margin-top: 6px;
    }

    /* Metrics table */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: .9rem;
      margin-top: 10px;
    }
    th {
      background: var(--purple);
      color: #fff;
      padding: 8px 12px;
      text-align: left;
    }
    td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
    tr:nth-child(even) td { background: var(--light-pur); }
    .active-row td { font-weight: bold; color: var(--wine); }

    footer {
      text-align: center;
      font-size: .78rem;
      color: #999;
      margin-top: 10px;
    }
  </style>
</head>
<body>

<header>
  <h1>🚗 Traffic Travel Time Predictor</h1>
  <p>Ridge &amp; Lasso Regression &mdash; trained on simulated traffic data</p>
</header>

<div class="card">
  <h2>Enter Traffic Features</h2>
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

    </div><!-- .grid -->

    <div class="btn-row">
      <button type="submit" name="model" value="ridge" class="btn-ridge">
        🔵 Predict with Ridge
      </button>
      <button type="submit" name="model" value="lasso" class="btn-lasso">
        🌸 Predict with Lasso
      </button>
    </div>
  </form>
</div><!-- .card -->

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

<footer>
  Run locally: <code>python traffic_predictor.py</code> &nbsp;|&nbsp;
  Deploy: set PORT env var and push to Heroku / Render / Railway
</footer>

</body>
</html>
"""

# ── 5. Flask routes ────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_used = None
    form       = {}

    if request.method == "POST":
        try:
            # Collect form values
            form = request.form.to_dict()
            tv  = float(form["traffic_volume"])
            spd = float(form["avg_speed"])
            dst = float(form["distance"])
            tod = float(form["time_of_day"])
            wthr = form.get("weather", "Clear")
            mdl  = form.get("model", "ridge")

            # Encode weather the same way the training encoder saw it
            wthr_enc = le.transform([wthr])[0]

            # Build and scale the feature vector
            raw_vec   = np.array([[tv, spd, dst, tod, wthr_enc]])
            scaled_vec = scaler.transform(raw_vec)

            # Choose model and predict
            if mdl == "lasso":
                pred_val   = lasso.predict(scaled_vec)[0]
                model_used = "Lasso"
            else:
                pred_val   = ridge.predict(scaled_vec)[0]
                model_used = "Ridge"

            prediction = round(pred_val, 2)

        except Exception as exc:
            # Surface error in prediction field for easy debugging
            prediction = f"Error: {exc}"

    metrics = {
        "ridge": {"mae": ridge_metrics[0], "rmse": ridge_metrics[1], "r2": ridge_metrics[2]},
        "lasso": {"mae": lasso_metrics[0], "rmse": lasso_metrics[1], "r2": lasso_metrics[2]},
    }

    return render_template_string(
        HTML,
        prediction     = prediction,
        model_used     = model_used,
        form           = form,
        weather_options= WEATHER_OPTIONS,
        metrics        = metrics,
    )


# ── 6. Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PORT env var is used by Heroku/Render; defaults to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

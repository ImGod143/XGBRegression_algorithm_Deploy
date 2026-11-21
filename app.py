"""
CareCost - Medical insurance charge estimator
=============================================
Flask web app that serves an XGBRegressor model (xgbregressor.pkl).

Model features (in order): age, sex, bmi, children, smoker, region

Run locally:
    pip install -r requirements.txt
    python app.py                 -> http://127.0.0.1:5000

Run in production:
    gunicorn -w 2 -b 0.0.0.0:8000 app:app

SECURITY NOTE: pickle files can execute code when loaded.
Only load model files that you created or fully trust.
"""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request

# --------------------------------------------------------------------------
# Configuration  (edit these if your training pipeline used other settings)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "xgbregressor.pkl"))

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

# Category encodings. These follow scikit-learn's LabelEncoder (alphabetical
# order), and the behaviour of the saved trees matches them (smoker=1 gives the
# much higher charge). If you encoded differently when training, change them here.
SEX_CODES = {"female": 0, "male": 1}
SMOKER_CODES = {"no": 0, "yes": 1}
REGION_CODES = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

# Input limits (the model was trained on adults aged 18-64)
LIMITS = {"age": (18, 64), "bmi": (15.0, 55.0), "children": (0, 5)}

# Display currency used by the web page (formatting only, no conversion)
CURRENCY = {"code": "USD", "locale": "en-US"}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "bmi": "BMI",
    "children": "Children",
    "smoker": "Smoking status",
    "region": "Region",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("carecost")

app = Flask(__name__)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
model = None
model_error = None
importance = []  # [{"name": "Smoking status", "share": 0.63}, ...]


def load_model():
    """Load the pickled model and compute normalised feature importance."""
    global model, model_error, importance
    try:
        with open(MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        model_error = None
        log.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any load problem to the UI
        model = None
        model_error = f"{type(exc).__name__}: {exc}"
        log.error("Could not load model: %s", model_error)
        return

    try:
        scores = model.get_booster().get_score(importance_type="total_gain")
        total = sum(scores.values()) or 1.0
        importance = sorted(
            (
                {"name": FEATURE_LABELS.get(k, k), "share": round(v / total, 4)}
                for k, v in scores.items()
            ),
            key=lambda item: item["share"],
            reverse=True,
        )
    except Exception as exc:  # noqa: BLE001 - importance is optional
        importance = []
        log.warning("Feature importance unavailable: %s", exc)


load_model()


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
def parse_payload(data):
    """Validate the JSON body. Returns (clean_dict, errors_dict)."""
    errors = {}
    clean = {}

    if not isinstance(data, dict):
        return None, {"error": "Send a JSON object."}

    try:
        age = int(data.get("age"))
        lo, hi = LIMITS["age"]
        if not lo <= age <= hi:
            raise ValueError
        clean["age"] = age
    except (TypeError, ValueError):
        errors["age"] = f"Age must be a whole number from {LIMITS['age'][0]} to {LIMITS['age'][1]}."

    sex = str(data.get("sex", "")).lower()
    if sex in SEX_CODES:
        clean["sex"] = sex
    else:
        errors["sex"] = "Choose female or male."

    try:
        bmi = float(data.get("bmi"))
        lo, hi = LIMITS["bmi"]
        if not lo <= bmi <= hi:
            raise ValueError
        clean["bmi"] = round(bmi, 2)
    except (TypeError, ValueError):
        errors["bmi"] = f"BMI must be between {LIMITS['bmi'][0]:g} and {LIMITS['bmi'][1]:g}."

    try:
        children = int(data.get("children"))
        lo, hi = LIMITS["children"]
        if not lo <= children <= hi:
            raise ValueError
        clean["children"] = children
    except (TypeError, ValueError):
        errors["children"] = f"Children must be a whole number from {LIMITS['children'][0]} to {LIMITS['children'][1]}."

    smoker = data.get("smoker")
    if isinstance(smoker, bool):
        smoker = "yes" if smoker else "no"
    smoker = str(smoker).lower()
    if smoker in SMOKER_CODES:
        clean["smoker"] = smoker
    else:
        errors["smoker"] = "Choose smoker or non-smoker."

    region = str(data.get("region", "")).lower()
    if region in REGION_CODES:
        clean["region"] = region
    else:
        errors["region"] = "Choose a region."

    return clean, errors


def encode_row(profile):
    """Turn a validated profile into one numeric model row."""
    return {
        "age": profile["age"],
        "sex": SEX_CODES[profile["sex"]],
        "bmi": profile["bmi"],
        "children": profile["children"],
        "smoker": SMOKER_CODES[profile["smoker"]],
        "region": REGION_CODES[profile["region"]],
    }


def predict_rows(rows):
    """Predict charges for a list of encoded rows in a single model call."""
    frame = pd.DataFrame(rows, columns=FEATURES).astype(
        {
            "age": "int64",
            "sex": "int64",
            "bmi": "float64",
            "children": "int64",
            "smoker": "int64",
            "region": "int64",
        }
    )
    return [max(float(v), 0.0) for v in model.predict(frame)]


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy weight"
    if bmi < 30:
        return "Overweight"
    return "Obesity"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    import json

    config = {
        "limits": LIMITS,
        "currency": CURRENCY,
        "regions": [{"value": k, "label": k.capitalize()} for k in REGION_CODES],
        "importance": importance,
        "modelReady": model is not None,
        "modelError": model_error,
    }
    html = INDEX_HTML.replace("__APP_CONFIG__", json.dumps(config))
    return Response(html, mimetype="text/html")


@app.post("/api/predict")
def api_predict():
    if model is None:
        return jsonify(error="The model is not loaded. Check that xgbregressor.pkl sits next to app.py."), 503

    profile, errors = parse_payload(request.get_json(silent=True))
    if errors:
        return jsonify(errors=errors, error=" ".join(errors.values())), 400

    user = encode_row(profile)
    ages = list(range(LIMITS["age"][0], LIMITS["age"][1] + 1))

    rows = [user]
    rows.append({**user, "smoker": SMOKER_CODES["no"]})
    rows.append({**user, "smoker": SMOKER_CODES["yes"]})
    rows += [{**user, "age": a, "smoker": SMOKER_CODES["no"]} for a in ages]
    rows += [{**user, "age": a, "smoker": SMOKER_CODES["yes"]} for a in ages]

    try:
        preds = predict_rows(rows)
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {exc}"), 500

    n = len(ages)
    estimate, non_smoker, smoker = preds[0], preds[1], preds[2]
    curve_no = preds[3 : 3 + n]
    curve_yes = preds[3 + n : 3 + 2 * n]

    return jsonify(
        estimate=round(estimate, 2),
        monthly=round(estimate / 12, 2),
        scenarios={"non_smoker": round(non_smoker, 2), "smoker": round(smoker, 2)},
        curve={
            "ages": ages,
            "non_smoker": [round(v, 2) for v in curve_no],
            "smoker": [round(v, 2) for v in curve_yes],
        },
        bmi_category=bmi_category(profile["bmi"]),
    )


@app.get("/healthz")
def healthz():
    return jsonify(status="ok" if model is not None else "model_not_loaded"), (200 if model is not None else 503)


# --------------------------------------------------------------------------
# Front end (single page, no external JS dependencies)
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CareCost | Health insurance charge estimator</title>
<meta name="description" content="Estimate yearly medical insurance charges from age, BMI, smoking status and more.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%230B7A85'/%3E%3Cpath d='M16 7v18M7 16h18' stroke='white' stroke-width='4' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Figtree:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#EDF2F4; --surface:#FFFFFF; --ink:#0D2B3A; --ink-2:#4F6677; --ink-3:#7F93A1;
    --line:#D6E1E6; --line-2:#E8EFF2;
    --brand:#0B7A85; --brand-deep:#096B75; --brand-soft:#DDF0F1;
    --warn:#D9482B; --warn-soft:#FCEAE5;
    --font-display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
    --font-body:"Figtree","Segoe UI",system-ui,-apple-system,sans-serif;
  }
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:16px;line-height:1.5;-webkit-font-smoothing:antialiased}
  button,input,select{font:inherit;color:inherit}
  .sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

  /* Header band */
  .band{background:var(--ink);color:#fff;padding:22px 24px 96px}
  .wrap{max-width:1180px;margin:0 auto}
  .brand{display:flex;align-items:center;gap:10px;font-family:var(--font-display);font-weight:700;font-size:20px;letter-spacing:-.01em}
  .brand svg{display:block}
  .band h1{font-family:var(--font-display);font-weight:700;font-size:clamp(30px,4.6vw,48px);line-height:1.05;letter-spacing:-.025em;margin:44px 0 12px;max-width:16ch}
  .band p{margin:0;color:#B7CAD5;max-width:52ch;font-size:17px}

  /* Layout */
  main{max-width:1180px;margin:-56px auto 0;padding:0 24px 56px;display:grid;grid-template-columns:minmax(0,430px) minmax(0,1fr);gap:24px;align-items:start}
  @media (max-width:959px){main{grid-template-columns:1fr}}
  @media (max-width:520px){.band{padding:18px 16px 84px}main{padding:0 12px 40px}}

  .card{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:28px}
  @media (max-width:520px){.card{padding:20px 18px}}
  .card h2{font-family:var(--font-display);font-size:22px;letter-spacing:-.01em;margin:0 0 4px}
  .card .hint{color:var(--ink-2);margin:0 0 22px;font-size:15px}

  /* Form */
  .field{margin-bottom:22px}
  .field-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
  .field label.lbl,.field .lbl{font-weight:600;font-size:15px}
  .readout{font-variant-numeric:tabular-nums;font-weight:600;color:var(--brand-deep)}
  .field-note{font-size:13px;color:var(--ink-3);margin:6px 0 0}
  .field-error{font-size:13px;color:var(--warn);margin:6px 0 0;min-height:0}

  .seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:4px;padding:4px;background:var(--line-2);border-radius:12px}
  .seg label{position:relative;display:block}
  .seg input{position:absolute;opacity:0;inset:0;margin:0;cursor:pointer}
  .seg span{display:block;text-align:center;padding:10px 12px;border-radius:9px;font-weight:600;color:var(--ink-2);transition:background .15s,color .15s,box-shadow .15s}
  .seg input:checked+span{background:#fff;color:var(--ink);box-shadow:0 1px 3px rgba(13,43,58,.16)}
  .seg.risk input[value="yes"]:checked+span{color:var(--warn)}
  .seg input:focus-visible+span{outline:2px solid var(--brand);outline-offset:2px}

  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:28px;margin:0;background:transparent;--p:50%}
  input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:99px;background:linear-gradient(to right,var(--brand) var(--p),var(--line) var(--p))}
  input[type=range]::-moz-range-track{height:6px;border-radius:99px;background:var(--line)}
  input[type=range]::-moz-range-progress{height:6px;border-radius:99px;background:var(--brand)}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;margin-top:-8px;border-radius:50%;background:#fff;border:2px solid var(--brand);box-shadow:0 1px 3px rgba(13,43,58,.25);cursor:grab}
  input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:#fff;border:2px solid var(--brand);cursor:grab}
  input[type=range]:focus{outline:none}
  input[type=range]:focus-visible::-webkit-slider-thumb{box-shadow:0 0 0 5px rgba(11,122,133,.25)}
  input[type=range]:focus-visible::-moz-range-thumb{box-shadow:0 0 0 5px rgba(11,122,133,.25)}

  .num{width:100%;padding:11px 14px;border:1px solid var(--line);border-radius:12px;background:#fff;font-variant-numeric:tabular-nums;font-weight:600}
  .num:focus-visible{outline:2px solid var(--brand);outline-offset:1px;border-color:var(--brand)}
  .num[aria-invalid=true]{border-color:var(--warn)}
  .bmi-row{display:grid;grid-template-columns:110px 1fr;gap:14px;align-items:center}
  .chip{display:inline-block;padding:3px 10px;border-radius:99px;font-size:13px;font-weight:600;background:var(--brand-soft);color:var(--brand-deep)}
  .chip.warn{background:var(--warn-soft);color:#A8321B}
  .chip.mid{background:#FBF1D9;color:#8A5A00}

  details.calc{margin-top:10px;border:1px dashed var(--line);border-radius:12px;padding:10px 14px}
  details.calc summary{cursor:pointer;font-weight:600;font-size:14px;color:var(--brand-deep)}
  details.calc summary:focus-visible{outline:2px solid var(--brand);outline-offset:3px;border-radius:6px}
  .calc-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
  .calc-grid label{font-size:13px;color:var(--ink-2);display:block;margin-bottom:4px}
  .btn-ghost{margin-top:12px;background:var(--brand-soft);color:var(--brand-deep);border:0;border-radius:10px;padding:9px 14px;font-weight:600;cursor:pointer}
  .btn-ghost:hover{background:#cde8ea}

  .stepper{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:12px;overflow:hidden}
  .stepper button{width:46px;height:44px;border:0;background:#fff;font-size:22px;line-height:1;cursor:pointer;color:var(--brand-deep)}
  .stepper button:hover:not(:disabled){background:var(--brand-soft)}
  .stepper button:disabled{color:var(--ink-3);opacity:.5;cursor:not-allowed}
  .stepper button:focus-visible{outline:2px solid var(--brand);outline-offset:-3px}
  .stepper output{min-width:48px;text-align:center;font-weight:700;font-variant-numeric:tabular-nums}

  .regions{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .regions label{position:relative;display:block}
  .regions input{position:absolute;opacity:0;inset:0;margin:0;cursor:pointer}
  .regions span{display:block;padding:11px 14px;border:1px solid var(--line);border-radius:12px;font-weight:600;color:var(--ink-2);transition:border-color .15s,background .15s,color .15s}
  .regions input:checked+span{border-color:var(--brand);background:var(--brand-soft);color:var(--brand-deep)}
  .regions input:focus-visible+span{outline:2px solid var(--brand);outline-offset:2px}

  .actions{display:flex;gap:12px;align-items:center;margin-top:8px}
  .btn{flex:1;border:0;border-radius:14px;padding:15px 20px;background:var(--brand);color:#fff;font-weight:700;font-size:16px;cursor:pointer;transition:background .15s,transform .05s}
  .btn:hover:not(:disabled){background:var(--brand-deep)}
  .btn:active:not(:disabled){transform:translateY(1px)}
  .btn:disabled{background:var(--ink-3);cursor:not-allowed}
  .btn:focus-visible{outline:3px solid rgba(11,122,133,.4);outline-offset:2px}
  .link-btn{background:none;border:0;color:var(--ink-2);font-weight:600;cursor:pointer;padding:10px 6px;text-decoration:underline;text-underline-offset:3px}
  .link-btn:hover{color:var(--ink)}

  /* Results */
  .results{display:grid;gap:20px;position:sticky;top:20px;transition:opacity .2s}
  @media (max-width:959px){.results{position:static}}
  .results.loading .amount,.results.loading .chart-wrap{opacity:.55}

  .alert{display:none;background:var(--warn-soft);color:#8E2A15;border:1px solid #F3C6BA;border-radius:14px;padding:12px 16px;font-weight:500}
  .alert.show{display:block}
  .alert code{background:rgba(0,0,0,.06);padding:1px 6px;border-radius:5px;font-size:.92em}

  .panel{background:var(--brand-deep);color:#fff;border-radius:20px;padding:30px 32px 28px;box-shadow:0 18px 40px -22px rgba(13,43,58,.6)}
  @media (max-width:520px){.panel{padding:24px 20px}}
  .panel .eyebrow{font-weight:600;color:#BFE6E9;font-size:15px}
  .amount{font-family:var(--font-display);font-weight:700;font-size:clamp(52px,8vw,84px);line-height:1;letter-spacing:-.035em;margin:10px 0 8px;font-variant-numeric:tabular-nums;transition:opacity .2s}
  .amount small{font-family:var(--font-body);font-size:18px;font-weight:500;letter-spacing:0;color:#BFE6E9;margin-left:8px}
  .panel .sub{color:#D7F0F2;font-size:17px;margin:0}
  .panel .context{margin:18px 0 0;padding-top:16px;border-top:1px solid rgba(255,255,255,.18);color:#BFE6E9;font-size:15px}

  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  @media (max-width:700px){.grid2{grid-template-columns:1fr}}
  .card.sm{padding:22px 24px}
  .card.sm h3{font-family:var(--font-display);font-size:18px;margin:0 0 4px;letter-spacing:-.01em}
  .card.sm .hint{font-size:14px;margin:0 0 16px}

  .bars{display:grid;gap:14px}
  .bar-row .top{display:flex;justify-content:space-between;align-items:baseline;font-size:14px;margin-bottom:6px;gap:8px}
  .bar-row .top b{font-variant-numeric:tabular-nums}
  .track{height:12px;border-radius:99px;background:var(--line-2);overflow:hidden}
  .fill{height:100%;border-radius:99px;background:var(--brand);width:0;transition:width .5s cubic-bezier(.2,.7,.2,1)}
  .fill.risk{background:var(--warn)}
  .fill.muted{background:#9DB7C2}
  .tag-you{display:inline-block;margin-left:6px;padding:1px 8px;border-radius:99px;background:var(--ink);color:#fff;font-size:12px;font-weight:600}
  .diff{margin:16px 0 0;font-size:14px;color:var(--ink-2)}

  /* Chart */
  .chart-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
  .legend{display:flex;gap:16px;font-size:14px;font-weight:600;color:var(--ink-2);flex-wrap:wrap}
  .legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
  .chart-wrap{position:relative;margin-top:8px;transition:opacity .2s;touch-action:pan-y}
  .chart-wrap svg{display:block;width:100%;height:auto;overflow:visible}
  .tip{position:absolute;top:0;pointer-events:none;background:var(--ink);color:#fff;border-radius:10px;padding:8px 12px;font-size:13px;line-height:1.45;white-space:nowrap;opacity:0;transform:translateX(-50%);transition:opacity .1s;z-index:2}
  .tip.on{opacity:1}
  .tip b{font-variant-numeric:tabular-nums}

  .drivers{display:grid;gap:12px}
  .drv{display:grid;grid-template-columns:104px 1fr 44px;gap:10px;align-items:center;font-size:14px}
  .drv .track{height:10px}
  .drv .fill{background:var(--ink)}
  .drv .pct{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink-2);font-weight:600}

  .fine{color:var(--ink-3);font-size:13px;margin:4px 4px 0}
  .spin{display:none;width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:-3px;margin-right:8px}
  .btn.busy .spin{display:inline-block}
  @keyframes spin{to{transform:rotate(360deg)}}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>

<header class="band">
  <div class="wrap">
    <div class="brand">
      <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#0B7A85"/><path d="M16 8v16M8 16h16" stroke="#fff" stroke-width="4" stroke-linecap="round"/></svg>
      CareCost
    </div>
    <h1>Estimate your yearly health insurance charge</h1>
    <p>Six details in, one number out. An XGBoost regression model trained on insurance billing records does the math.</p>
  </div>
</header>

<main>
  <!-- ============ FORM ============ -->
  <section class="card" aria-labelledby="form-title">
    <h2 id="form-title">Your profile</h2>
    <p class="hint">Results update as you change a value.</p>

    <form id="form" novalidate autocomplete="off">
      <div class="field">
        <div class="field-head">
          <label class="lbl" for="age">Age</label>
          <span class="readout" id="ageOut">35 years</span>
        </div>
        <input type="range" id="age" name="age" min="18" max="64" step="1" value="35">
      </div>

      <div class="field">
        <div class="field-head"><span class="lbl" id="sexLbl">Sex</span></div>
        <div class="seg" role="radiogroup" aria-labelledby="sexLbl">
          <label><input type="radio" name="sex" value="female" checked><span>Female</span></label>
          <label><input type="radio" name="sex" value="male"><span>Male</span></label>
        </div>
      </div>

      <div class="field">
        <div class="field-head">
          <label class="lbl" for="bmi">Body mass index (BMI)</label>
          <span class="chip" id="bmiChip">Overweight</span>
        </div>
        <div class="bmi-row">
          <input class="num" type="number" id="bmi" name="bmi" min="15" max="55" step="0.1" value="26.5" inputmode="decimal" aria-describedby="bmiErr">
          <input type="range" id="bmiRange" name="bmiRange" min="15" max="55" step="0.1" value="26.5" aria-label="BMI slider">
        </div>
        <p class="field-error" id="bmiErr" role="alert"></p>
        <details class="calc">
          <summary>Don't know your BMI? Calculate it</summary>
          <div class="calc-grid">
            <div><label for="hCm">Height (cm)</label><input class="num" type="number" id="hCm" min="100" max="230" step="1" placeholder="170" inputmode="numeric"></div>
            <div><label for="wKg">Weight (kg)</label><input class="num" type="number" id="wKg" min="30" max="250" step="0.5" placeholder="70" inputmode="decimal"></div>
          </div>
          <button class="btn-ghost" type="button" id="useBmi">Use this BMI</button>
          <span class="field-note" id="calcOut" style="margin-left:10px"></span>
        </details>
      </div>

      <div class="field">
        <div class="field-head"><span class="lbl" id="childLbl">Children covered</span></div>
        <div class="stepper" role="group" aria-labelledby="childLbl">
          <button type="button" id="childMinus" aria-label="Fewer children">&minus;</button>
          <output id="childOut" aria-live="polite">0</output>
          <button type="button" id="childPlus" aria-label="More children">+</button>
        </div>
      </div>

      <div class="field">
        <div class="field-head"><span class="lbl" id="smokeLbl">Smoking</span></div>
        <div class="seg risk" role="radiogroup" aria-labelledby="smokeLbl">
          <label><input type="radio" name="smoker" value="no" checked><span>Non-smoker</span></label>
          <label><input type="radio" name="smoker" value="yes"><span>Smoker</span></label>
        </div>
      </div>

      <div class="field">
        <div class="field-head"><span class="lbl" id="regionLbl">Region</span></div>
        <div class="regions" role="radiogroup" aria-labelledby="regionLbl" id="regions"></div>
      </div>

      <div class="actions">
        <button class="btn" type="submit" id="go"><span class="spin" aria-hidden="true"></span>Estimate charge</button>
        <button class="link-btn" type="button" id="reset">Reset</button>
      </div>
    </form>
  </section>

  <!-- ============ RESULTS ============ -->
  <section class="results" id="results" aria-labelledby="res-title">
    <div class="alert" id="alert" role="alert"></div>

    <div class="panel">
      <div class="eyebrow" id="res-title">Estimated yearly charge</div>
      <div class="amount" id="amount">&nbsp;<small id="cur"></small></div>
      <p class="sub" id="monthly">&nbsp;</p>
      <p class="context" id="context">&nbsp;</p>
    </div>
    <div class="sr-only" id="live" aria-live="polite"></div>

    <div class="card sm">
      <div class="chart-head">
        <div>
          <h3>How the charge changes with age</h3>
          <p class="hint">Everything else stays as you entered it.</p>
        </div>
        <div class="legend">
          <span><i style="background:var(--brand)"></i>Non-smoker</span>
          <span><i style="background:var(--warn)"></i>Smoker</span>
        </div>
      </div>
      <div class="chart-wrap" id="chart"></div>
    </div>

    <div class="grid2">
      <div class="card sm">
        <h3>Smoking status</h3>
        <p class="hint">Same profile, both options.</p>
        <div class="bars" id="bars"></div>
        <p class="diff" id="diff"></p>
      </div>
      <div class="card sm">
        <h3>What the model weighs most</h3>
        <p class="hint">Share of the model's total learning gain.</p>
        <div class="drivers" id="drivers"></div>
      </div>
    </div>

    <p class="fine">This is a statistical estimate, not an insurance quote. Real premiums depend on the insurer, plan and underwriting.</p>
  </section>
</main>

<script>
(() => {
  const CFG = __APP_CONFIG__;
  const L = CFG.limits;
  const $ = (s, r = document) => r.querySelector(s);
  const money = new Intl.NumberFormat(CFG.currency.locale, {style:'currency', currency:CFG.currency.code, maximumFractionDigits:0});
  const compact = new Intl.NumberFormat(CFG.currency.locale, {style:'currency', currency:CFG.currency.code, notation:'compact', maximumFractionDigits:1});
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const DEFAULTS = {age:35, sex:'female', bmi:26.5, children:0, smoker:'no', region:CFG.regions[0].value};
  const state = {...DEFAULTS};
  let bmiOk = true;

  const form = $('#form'), results = $('#results'), goBtn = $('#go');
  const regionLabel = v => (CFG.regions.find(r => r.value === v) || {}).label || v;

  /* ---------- build region radios ---------- */
  $('#regions').innerHTML = CFG.regions.map((r, i) =>
    `<label><input type="radio" name="region" value="${r.value}"${i === 0 ? ' checked' : ''}><span>${r.label}</span></label>`
  ).join('');

  /* ---------- build "what the model weighs" ---------- */
  const drivers = $('#drivers');
  if (CFG.importance.length) {
    drivers.innerHTML = CFG.importance.map(d =>
      `<div class="drv"><span>${d.name}</span><div class="track"><div class="fill" style="width:${(d.share * 100).toFixed(1)}%"></div></div><span class="pct">${d.share >= 0.01 ? Math.round(d.share * 100) : '<1'}%</span></div>`
    ).join('');
  } else {
    drivers.closest('.card').style.display = 'none';
  }

  /* ---------- helpers ---------- */
  const setRange = el => {
    const min = +el.min, max = +el.max;
    el.style.setProperty('--p', ((el.value - min) / (max - min) * 100) + '%');
  };

  function bmiInfo(b) {
    if (b < 18.5) return ['Underweight', 'mid'];
    if (b < 25) return ['Healthy weight', ''];
    if (b < 30) return ['Overweight', 'mid'];
    return ['Obesity', 'warn'];
  }

  function paintBmi() {
    const chip = $('#bmiChip');
    if (!bmiOk) { chip.textContent = 'Check value'; chip.className = 'chip warn'; return; }
    const [t, c] = bmiInfo(state.bmi);
    chip.textContent = t; chip.className = 'chip ' + c;
  }

  function paintChildren() {
    $('#childOut').textContent = state.children;
    $('#childMinus').disabled = state.children <= L.children[0];
    $('#childPlus').disabled = state.children >= L.children[1];
  }

  function applyState() {
    $('#age').value = state.age; setRange($('#age')); $('#ageOut').textContent = state.age + ' years';
    form.elements.sex.value = state.sex;
    form.elements.smoker.value = state.smoker;
    form.elements.region.value = state.region;
    $('#bmi').value = state.bmi; $('#bmiRange').value = state.bmi; setRange($('#bmiRange'));
    $('#bmi').removeAttribute('aria-invalid'); $('#bmiErr').textContent = ''; bmiOk = true;
    paintBmi(); paintChildren();
  }

  /* ---------- events ---------- */
  let timer;
  const schedule = () => { clearTimeout(timer); timer = setTimeout(run, 220); };

  form.addEventListener('input', e => {
    const t = e.target;
    if (t.id === 'hCm' || t.id === 'wKg') return;
    switch (t.name) {
      case 'age':
        state.age = +t.value; setRange(t); $('#ageOut').textContent = t.value + ' years'; break;
      case 'sex': case 'smoker': case 'region':
        state[t.name] = t.value; break;
      case 'bmiRange':
        state.bmi = +(+t.value).toFixed(1); $('#bmi').value = state.bmi; setRange(t);
        bmiOk = true; $('#bmi').removeAttribute('aria-invalid'); $('#bmiErr').textContent = ''; paintBmi(); break;
      case 'bmi': {
        const v = parseFloat(t.value);
        if (Number.isFinite(v) && v >= L.bmi[0] && v <= L.bmi[1]) {
          bmiOk = true; state.bmi = v; t.removeAttribute('aria-invalid'); $('#bmiErr').textContent = '';
          $('#bmiRange').value = v; setRange($('#bmiRange'));
        } else {
          bmiOk = false; t.setAttribute('aria-invalid', 'true');
          $('#bmiErr').textContent = `Enter a BMI between ${L.bmi[0]} and ${L.bmi[1]}.`;
        }
        paintBmi(); break;
      }
    }
    if (bmiOk) schedule();
  });

  $('#childMinus').addEventListener('click', () => { state.children = Math.max(L.children[0], state.children - 1); paintChildren(); schedule(); });
  $('#childPlus').addEventListener('click', () => { state.children = Math.min(L.children[1], state.children + 1); paintChildren(); schedule(); });

  $('#useBmi').addEventListener('click', () => {
    const h = parseFloat($('#hCm').value), w = parseFloat($('#wKg').value), out = $('#calcOut');
    if (!(h >= 100 && h <= 230 && w >= 30 && w <= 250)) { out.textContent = 'Enter height 100-230 cm and weight 30-250 kg.'; return; }
    const b = w / Math.pow(h / 100, 2);
    state.bmi = Math.min(L.bmi[1], Math.max(L.bmi[0], +b.toFixed(1)));
    out.textContent = 'BMI set to ' + state.bmi;
    applyState(); run();
  });

  form.addEventListener('submit', e => {
    e.preventDefault();
    clearTimeout(timer);
    run(true);
  });

  $('#reset').addEventListener('click', () => { Object.assign(state, DEFAULTS); applyState(); run(); });

  /* ---------- API call ---------- */
  let ctrl, seq = 0, last = null;

  async function run(scroll) {
    if (!bmiOk || !CFG.modelReady) return;
    ctrl && ctrl.abort(); ctrl = new AbortController();
    const mine = ++seq;
    results.classList.add('loading'); goBtn.classList.add('busy');
    try {
      const res = await fetch('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(state), signal: ctrl.signal
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || 'The estimate could not be calculated.');
      if (mine !== seq) return;
      last = {...state}; render(data, last); showError('');
      if (scroll === true && matchMedia('(max-width:959px)').matches) results.scrollIntoView({behavior: reduce ? 'auto' : 'smooth', block:'start'});
    } catch (err) {
      if (err.name === 'AbortError') return;
      showError(err.message);
    } finally {
      if (mine === seq) { results.classList.remove('loading'); goBtn.classList.remove('busy'); }
    }
  }

  function showError(msg) {
    const a = $('#alert');
    a.textContent = msg; a.classList.toggle('show', !!msg);
  }

  /* ---------- rendering ---------- */
  let shown = 0, raf;
  function countTo(to) {
    const el = $('#amount'); cancelAnimationFrame(raf);
    const paint = v => { el.firstChild.nodeValue = money.format(v); };
    if (reduce) { shown = to; paint(to); return; }
    const from = shown, t0 = performance.now(), dur = 550;
    const tick = now => {
      const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      shown = from + (to - from) * e; paint(shown);
      if (p < 1) raf = requestAnimationFrame(tick); else { shown = to; paint(to); }
    };
    raf = requestAnimationFrame(tick);
  }

  function render(d, s) {
    $('#amount').innerHTML = '0<small id="cur"></small>';
    $('#cur').textContent = CFG.currency.code + ' / year';
    countTo(d.estimate);
    $('#monthly').textContent = 'About ' + money.format(d.monthly) + ' per month';
    const kids = s.children === 0 ? 'no children' : s.children === 1 ? '1 child' : s.children + ' children';
    $('#context').textContent =
      `For a ${s.age}-year-old ${s.sex} ${s.smoker === 'yes' ? 'smoker' : 'non-smoker'} in the ${regionLabel(s.region)}, BMI ${s.bmi}, with ${kids}.`;
    $('#live').textContent = `Estimated yearly charge ${money.format(d.estimate)}, about ${money.format(d.monthly)} per month.`;

    renderBars(d, s);
    renderChart(d, s);
  }

  function renderBars(d, s) {
    const no = d.scenarios.non_smoker, yes = d.scenarios.smoker, max = Math.max(no, yes) || 1;
    const you = s.smoker === 'yes' ? 'yes' : 'no';
    const row = (key, label, val, cls) => `
      <div class="bar-row">
        <div class="top"><span>${label}${you === key ? '<span class="tag-you">You</span>' : ''}</span><b>${money.format(val)}</b></div>
        <div class="track"><div class="fill ${you === key ? cls : 'muted'}" data-w="${(val / max * 100).toFixed(1)}"></div></div>
      </div>`;
    const box = $('#bars');
    box.innerHTML = row('no', 'Non-smoker', no, '') + row('yes', 'Smoker', yes, 'risk');
    requestAnimationFrame(() => box.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%'));
    const gap = Math.abs(yes - no);
    $('#diff').textContent = `The model estimates ${money.format(gap)} ${yes >= no ? 'more' : 'less'} per year for a smoker with this profile.`;
  }

  /* ---------- chart ---------- */
  const M = {l:48, r:12, t:14, b:30};
  let W = 640, H = 300, IW = 0, IH = 0;
  let geo = null, lastRender = null;

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }

  function renderChart(d, s) {
    lastRender = [d, s];
    W = Math.max(300, Math.round($('#chart').clientWidth || 640));
    H = W < 480 ? 250 : 300;
    IW = W - M.l - M.r; IH = H - M.t - M.b;
    const ages = d.curve.ages, a0 = ages[0], a1 = ages[ages.length - 1];
    const peak = Math.max(...d.curve.smoker, ...d.curve.non_smoker);
    const step = niceStep(peak / 4), ymax = Math.ceil(peak / step) * step;
    const x = a => M.l + (a - a0) / (a1 - a0) * IW;
    const y = v => M.t + IH - (v / ymax) * IH;
    const path = arr => arr.map((v, i) => (i ? 'L' : 'M') + x(ages[i]).toFixed(1) + ' ' + y(v).toFixed(1)).join('');

    let grid = '';
    for (let v = 0; v <= ymax + 1e-6; v += step) {
      grid += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}" stroke="${v === 0 ? '#9DB7C2' : '#E3ECF0'}"/>` +
              `<text x="${M.l - 10}" y="${y(v) + 4}" text-anchor="end" font-size="12" fill="#7F93A1">${compact.format(v)}</text>`;
    }
    let xt = '';
    [20, 30, 40, 50, 60].forEach(a => {
      if (a < a0 || a > a1) return;
      xt += `<text x="${x(a)}" y="${H - 10}" text-anchor="middle" font-size="12" fill="#7F93A1">${a}</text>`;
    });

    const own = s.smoker === 'yes' ? '#D9482B' : '#0B7A85';
    const ax = x(Math.min(a1, Math.max(a0, s.age)));

    $('#chart').innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Estimated yearly charge by age for smokers and non-smokers. Your estimate at age ${s.age} is ${money.format(d.estimate)}.">
        ${grid}${xt}
        <path d="${path(d.curve.non_smoker)}" fill="none" stroke="#0B7A85" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="${path(d.curve.smoker)}" fill="none" stroke="#D9482B" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        <line x1="${ax}" x2="${ax}" y1="${M.t}" y2="${M.t + IH}" stroke="${own}" stroke-width="1.5" stroke-dasharray="4 4" opacity=".55"/>
        <circle cx="${ax}" cy="${y(d.estimate)}" r="7" fill="${own}" stroke="#fff" stroke-width="3"/>
        <g id="hov" style="display:none">
          <line id="hovLine" y1="${M.t}" y2="${M.t + IH}" stroke="#0D2B3A" stroke-width="1" opacity=".35"/>
          <circle id="hovA" r="4.5" fill="#0B7A85" stroke="#fff" stroke-width="2"/>
          <circle id="hovB" r="4.5" fill="#D9482B" stroke="#fff" stroke-width="2"/>
        </g>
        <rect id="hit" x="${M.l}" y="${M.t}" width="${IW}" height="${IH}" fill="transparent"/>
      </svg>
      <div class="tip" id="tip"></div>`;

    geo = {d, ages, a0, a1, x, y};
  }

  const chart = $('#chart');
  let rz;
  addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (lastRender && Math.abs(chart.clientWidth - W) > 4) renderChart(...lastRender); }, 150);
  });
  function hover(e) {
    if (!geo) return;
    const svg = chart.querySelector('svg'); if (!svg) return;
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width * W;
    let age = Math.round(geo.a0 + (px - M.l) / IW * (geo.a1 - geo.a0));
    age = Math.min(geo.a1, Math.max(geo.a0, age));
    const i = age - geo.a0, no = geo.d.curve.non_smoker[i], yes = geo.d.curve.smoker[i];
    const cx = geo.x(age);
    $('#hov').style.display = '';
    $('#hovLine').setAttribute('x1', cx); $('#hovLine').setAttribute('x2', cx);
    $('#hovA').setAttribute('cx', cx); $('#hovA').setAttribute('cy', geo.y(no));
    $('#hovB').setAttribute('cx', cx); $('#hovB').setAttribute('cy', geo.y(yes));
    const tip = $('#tip');
    tip.innerHTML = `Age <b>${age}</b><br>Non-smoker <b>${money.format(no)}</b><br>Smoker <b>${money.format(yes)}</b>`;
    const w = tip.offsetWidth, cw = r.width;
    let left = cx / W * cw; left = Math.min(cw - w / 2, Math.max(w / 2, left));
    tip.style.left = left + 'px'; tip.classList.add('on');
  }
  chart.addEventListener('pointermove', hover);
  chart.addEventListener('pointerdown', hover);
  chart.addEventListener('pointerleave', () => {
    const h = $('#hov'); if (h) h.style.display = 'none';
    const t = $('#tip'); if (t) t.classList.remove('on');
  });

  /* ---------- boot ---------- */
  applyState();
  if (!CFG.modelReady) {
    goBtn.disabled = true;
    const a = $('#alert');
    a.innerHTML = 'The model file could not be loaded. Place <code>xgbregressor.pkl</code> next to <code>app.py</code> and restart the server.';
    a.classList.add('show');
    $('#amount').innerHTML = '&mdash;<small></small>';
  } else {
    run();
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )

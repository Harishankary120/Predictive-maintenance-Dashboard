"""
ml_model.py
------------
Loads the trained RandomForest failure model and exposes a single clean
function, predict_failure(telemetry, history), used by main.py (Feature 5).

If models/failure_model.pkl hasn't been trained yet (or fails to load),
this falls back to a transparent rule-based probability estimate instead
of crashing the backend, so the API/dashboard keep working before the
model has been trained.
"""

import os
import numpy as np
import pandas as pd
import joblib

import config

_model_bundle = None
_load_attempted = False


def _load_model():
    global _model_bundle, _load_attempted
    if _load_attempted:
        return _model_bundle
    _load_attempted = True
    if os.path.exists(config.MODEL_PATH):
        try:
            _model_bundle = joblib.load(config.MODEL_PATH)
            print(f"ML model loaded from {config.MODEL_PATH}")
        except Exception as e:
            print(f"Could not load ML model ({e}). Falling back to rule-based prediction.")
            _model_bundle = None
    else:
        print(f"No trained model found at {config.MODEL_PATH}. Falling back to rule-based prediction.")
        _model_bundle = None
    return _model_bundle


def _engineer_features(telemetry: dict, history: list) -> dict:
    """
    history: list of recent readings (oldest->newest), each a dict with at
    least temperature/rpm/vibration/is_anomaly, as maintained by main.py.
    """
    temps = [h["temperature"] for h in history] + [telemetry["temperature"]]
    rpms = [h["rpm"] for h in history] + [telemetry["rpm"]]
    vibs = [h["vibration"] for h in history] + [telemetry["vibration"]]
    anomalies = [1 if h.get("is_anomaly") else 0 for h in history]

    def slope(values):
        if len(values) < 2:
            return 0.0
        return float(np.polyfit(range(len(values)), values, 1)[0])

    return {
        "temperature": telemetry["temperature"],
        "rpm": telemetry["rpm"],
        "vibration": telemetry["vibration"],
        "temp_trend": slope(temps),
        "rpm_trend": slope(rpms),
        "vibration_trend": slope(vibs),
        "temp_rolling_avg": float(np.mean(temps)),
        "rpm_rolling_avg": float(np.mean(rpms)),
        "vibration_rolling_avg": float(np.mean(vibs)),
        "vibration_rolling_std": float(np.std(vibs)),
        "recent_anomaly_count": int(sum(anomalies)),
    }


def _rule_based_probability(features: dict) -> float:
    """Transparent fallback used only if the trained model isn't available."""
    t_low, t_high = config.NORMAL_TEMP_RANGE
    r_low, r_high = config.NORMAL_RPM_RANGE
    v_low, v_high = config.NORMAL_VIBRATION_RANGE

    score = 0.0
    score += max(0.0, (features["temperature"] - t_high) / (config.DANGER_TEMP - t_high)) * 0.3
    score += max(0.0, (features["vibration"] - v_high) / (config.DANGER_VIBRATION - v_high)) * 0.4
    if features["rpm"] < r_low:
        score += max(0.0, (r_low - features["rpm"]) / (r_low - config.DANGER_RPM_LOW)) * 0.15
    elif features["rpm"] > r_high:
        score += min(1.0, (features["rpm"] - r_high) / 100.0) * 0.15
    score += min(1.0, features["recent_anomaly_count"] / 5.0) * 0.15

    return float(min(1.0, max(0.0, score)))


def predict_failure(telemetry: dict, history: list) -> dict:
    """
    telemetry: {"temperature": float, "rpm": float, "vibration": float}
    history:   list of recent readings for this machine (oldest->newest)

    Returns {"failure_probability": float, "failure_risk": str, "features": dict}
    """
    features = _engineer_features(telemetry, history)
    bundle = _load_model()

    if bundle is not None:
        model = bundle["model"]
        feature_order = bundle["features"]
        X = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)
        try:
            probability = float(model.predict_proba(X)[0][1])
        except Exception as e:
            print(f"ML prediction failed ({e}), falling back to rule-based estimate.")
            probability = _rule_based_probability(features)
    else:
        probability = _rule_based_probability(features)

    return {
        "failure_probability": round(probability, 4),
        "failure_risk": config.risk_category(probability),
        "features": features,
    }

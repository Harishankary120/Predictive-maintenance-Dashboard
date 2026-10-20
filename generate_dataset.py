"""
generate_dataset.py
--------------------
Generates a synthetic manufacturing telemetry dataset for training the
failure-prediction model (Feature 3).

IMPORTANT: This is synthetic data meant to demonstrate an end-to-end ML
pipeline for a prototype/demo. It is NOT real sensor data and the trained
model should not be presented as having real-world industrial accuracy.

Assumptions (documented so the synthetic data can be reasoned about):
- Normal operation: temperature 60-75C, RPM 1180-1220, vibration 0.02-0.08g,
  matching the ranges already used by simulator.py / health.py / config.py.
- Each row represents ONE snapshot of a machine plus a short rolling window
  of history, already reduced to engineered features (trend, rolling mean,
  rolling std, recent anomaly count) - this mirrors exactly what
  ml_model.py computes at inference time from main.py's history buffer, so
  the training features line up with the live features.
- "failure" = 1 means the machine is in a state that, if left unattended,
  is heading toward a breakdown in the near term (i.e. failure-risk
  condition), not necessarily "already broken".
- Six scenario families are generated in roughly equal proportion:
    1. healthy operation                (label 0)
    2. overheating                      (label 1)
    3. excessive vibration              (label 1)
    4. abnormal RPM                     (label 1)
    5. combined degradation             (label 1)
    6. borderline/noisy healthy         (label 0) - keeps the model honest
       instead of learning "any noise = failure"
"""

import numpy as np
import pandas as pd
import os

import config

RNG = np.random.default_rng(42)

N_PER_SCENARIO = 1500  # rows per scenario -> balanced-ish dataset


def _clip(values, lo, hi):
    return np.clip(values, lo, hi)


def _rolling_features(temp, rpm, vib, anomaly_flags):
    """Given arrays representing a short recent history (oldest->newest),
    compute the same engineered features ml_model.py computes live."""
    temp_trend = np.polyfit(range(len(temp)), temp, 1)[0] if len(temp) > 1 else 0.0
    rpm_trend = np.polyfit(range(len(rpm)), rpm, 1)[0] if len(rpm) > 1 else 0.0
    vib_trend = np.polyfit(range(len(vib)), vib, 1)[0] if len(vib) > 1 else 0.0
    return {
        "temperature": temp[-1],
        "rpm": rpm[-1],
        "vibration": vib[-1],
        "temp_trend": temp_trend,
        "rpm_trend": rpm_trend,
        "vibration_trend": vib_trend,
        "temp_rolling_avg": float(np.mean(temp)),
        "rpm_rolling_avg": float(np.mean(rpm)),
        "vibration_rolling_avg": float(np.mean(vib)),
        "vibration_rolling_std": float(np.std(vib)),
        "recent_anomaly_count": int(np.sum(anomaly_flags)),
    }


def _make_window(temp_center, temp_noise, rpm_center, rpm_noise, vib_center, vib_noise,
                  temp_drift=0.0, rpm_drift=0.0, vib_drift=0.0, window=10, anomaly_prob=0.0):
    idx = np.arange(window)
    temp = temp_center + temp_drift * idx + RNG.normal(0, temp_noise, window)
    rpm = rpm_center + rpm_drift * idx + RNG.normal(0, rpm_noise, window)
    vib = vib_center + vib_drift * idx + RNG.normal(0, vib_noise, window)
    vib = _clip(vib, 0.0, 1.0)
    rpm = _clip(rpm, 0.0, 2500.0)
    temp = _clip(temp, 0.0, 200.0)
    anomalies = (RNG.random(window) < anomaly_prob).astype(int)
    return temp, rpm, vib, anomalies


def generate():
    rows = []

    # 1. Healthy operation -----------------------------------------------
    for _ in range(N_PER_SCENARIO):
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(63, 72), temp_noise=1.0,
            rpm_center=RNG.uniform(1190, 1210), rpm_noise=4,
            vib_center=RNG.uniform(0.03, 0.06), vib_noise=0.005,
            anomaly_prob=0.0,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 0
        rows.append(feats)

    # 2. Overheating --------------------------------------------------------
    for _ in range(N_PER_SCENARIO):
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(78, 86), temp_noise=1.5, temp_drift=RNG.uniform(0.3, 1.2),
            rpm_center=RNG.uniform(1150, 1210), rpm_noise=6,
            vib_center=RNG.uniform(0.05, 0.09), vib_noise=0.006,
            anomaly_prob=0.15,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 1
        rows.append(feats)

    # 3. Excessive vibration (bearing wear / imbalance) ----------------------
    for _ in range(N_PER_SCENARIO):
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(68, 78), temp_noise=1.2,
            rpm_center=RNG.uniform(1150, 1220), rpm_noise=8,
            vib_center=RNG.uniform(0.12, 0.22), vib_noise=0.02, vib_drift=RNG.uniform(0.001, 0.01),
            anomaly_prob=0.35,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 1
        rows.append(feats)

    # 4. Abnormal RPM (motor instability) ------------------------------------
    for _ in range(N_PER_SCENARIO):
        low_side = RNG.random() < 0.5
        rpm_center = RNG.uniform(880, 970) if low_side else RNG.uniform(1260, 1350)
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(68, 80), temp_noise=1.5,
            rpm_center=rpm_center, rpm_noise=25,
            vib_center=RNG.uniform(0.06, 0.12), vib_noise=0.01,
            anomaly_prob=0.2,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 1
        rows.append(feats)

    # 5. Combined degradation (worst case - everything sliding at once) -----
    for _ in range(N_PER_SCENARIO):
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(85, 100), temp_noise=2.0, temp_drift=RNG.uniform(0.5, 1.5),
            rpm_center=RNG.uniform(870, 960), rpm_noise=20,
            vib_center=RNG.uniform(0.18, 0.32), vib_noise=0.03, vib_drift=RNG.uniform(0.005, 0.02),
            anomaly_prob=0.6,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 1
        rows.append(feats)

    # 6. Borderline / noisy but still healthy (keeps the model from treating
    #    any noise as failure) ----------------------------------------------
    for _ in range(N_PER_SCENARIO):
        temp, rpm, vib, anom = _make_window(
            temp_center=RNG.uniform(73, 77), temp_noise=2.5,
            rpm_center=RNG.uniform(1160, 1230), rpm_noise=12,
            vib_center=RNG.uniform(0.07, 0.10), vib_noise=0.015,
            anomaly_prob=0.05,
        )
        feats = _rolling_features(temp, rpm, vib, anom)
        feats["failure"] = 0
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(config.TRAINING_DATA_PATH), exist_ok=True)
    df.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"Generated {len(df)} rows -> {config.TRAINING_DATA_PATH}")
    print(df["failure"].value_counts())

"""
health.py
---------
Deterministic, explainable Machine Health Score (0-100) and the
"why is this machine at risk" reasoning used by the dashboard.

Nothing here is random. The score is a weighted deduction from 100 based
on how far each sensor reading sits outside its normal operating range,
whether the machine's recent trend is worsening, and how many anomalies
have fired recently. This keeps the score reproducible and easy to explain
to a judge/reviewer during a demo.
"""

import config


def _range_penalty(value, normal_range, danger_value, max_penalty, danger_is_low=False):
    """
    Returns a 0..max_penalty deduction based on how far `value` sits
    outside `normal_range`. 0 if inside the normal range, scaling up to
    max_penalty once `value` reaches/passes the danger threshold.
    """
    low, high = normal_range

    if low <= value <= high:
        return 0.0

    if value > high:
        span = max(danger_value - high, 1e-6)
        overshoot = min(value - high, span)
        return max_penalty * (overshoot / span)

    # value < low
    if danger_is_low:
        span = max(low - danger_value, 1e-6)
        undershoot = min(low - value, span)
        return max_penalty * (undershoot / span)

    # No explicit "danger low" configured (e.g. temperature) - use a
    # symmetric fallback so being far below the range still costs points.
    span = max(high - low, 1e-6)
    undershoot = min(low - value, span)
    return max_penalty * (undershoot / span)


def calculate_health_score(telemetry: dict, trends: dict, recent_anomaly_count: int) -> dict:
    """
    telemetry: {"temperature": float, "rpm": float, "vibration": float}
    trends:    {"temperature": slope, "rpm": slope, "vibration": slope}
    recent_anomaly_count: number of anomalies in the recent history window

    Returns {"health_score": int 0-100, "health_status": "HEALTHY"|"WARNING"|"CRITICAL"}
    """
    score = 100.0

    # --- Sensor-range deductions (weighted by how safety-critical each is) ---
    score -= _range_penalty(
        telemetry["temperature"], config.NORMAL_TEMP_RANGE, config.DANGER_TEMP, max_penalty=30
    )
    score -= _range_penalty(
        telemetry["vibration"], config.NORMAL_VIBRATION_RANGE, config.DANGER_VIBRATION, max_penalty=35
    )
    score -= _range_penalty(
        telemetry["rpm"], config.NORMAL_RPM_RANGE, config.DANGER_RPM_LOW, max_penalty=20, danger_is_low=True
    )

    # --- Trend penalty: worsening direction costs extra even if still in range ---
    # Positive slope on temperature/vibration = getting worse.
    if trends.get("temperature", 0) > 0.05:
        score -= min(8, trends["temperature"] * 2)
    if trends.get("vibration", 0) > 0.001:
        score -= min(10, trends["vibration"] * 400)
    if abs(trends.get("rpm", 0)) > 3:
        score -= min(7, abs(trends["rpm"]) * 0.5)

    # --- Anomaly frequency penalty ---
    score -= min(15, recent_anomaly_count * 3)

    score = max(0, min(100, round(score)))

    if score >= config.HEALTH_HEALTHY_MIN:
        status = "HEALTHY"
    elif score >= config.HEALTH_WARNING_MIN:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {"health_score": int(score), "health_status": status}


def generate_reasons(telemetry: dict, trends: dict, recent_anomaly_count: int) -> list:
    """
    Builds a list of plain-English reasons grounded in the actual telemetry
    values - only reasons that genuinely apply are included (Feature 10 /
    Feature 17). Used for both the failure-prediction explanation and the
    "Why is this machine at risk?" dashboard panel.
    """
    reasons = []
    temp = telemetry["temperature"]
    rpm = telemetry["rpm"]
    vib = telemetry["vibration"]

    t_low, t_high = config.NORMAL_TEMP_RANGE
    r_low, r_high = config.NORMAL_RPM_RANGE
    v_low, v_high = config.NORMAL_VIBRATION_RANGE

    if vib > v_high:
        reasons.append(
            f"Vibration ({vib:.4f} g) is above the normal range ({v_low:.2f}-{v_high:.2f} g)"
        )
    if trends.get("vibration", 0) > 0.001:
        reasons.append("Vibration has been trending upward over recent readings")

    if temp > t_high:
        reasons.append(
            f"Temperature ({temp:.1f}\u00b0C) is above the normal range ({t_low:.0f}-{t_high:.0f}\u00b0C)"
        )
    if trends.get("temperature", 0) > 0.05:
        reasons.append("Temperature has been rising over recent readings")

    if rpm < r_low or rpm > r_high:
        reasons.append(
            f"RPM ({rpm:.0f}) is outside the normal operating range ({r_low:.0f}-{r_high:.0f})"
        )
    if abs(trends.get("rpm", 0)) > 3:
        reasons.append("RPM has become unstable over recent readings")

    if recent_anomaly_count >= 3:
        reasons.append(f"{recent_anomaly_count} anomalies detected in the recent telemetry window")
    elif recent_anomaly_count >= 1:
        reasons.append("A recent anomaly was detected in the telemetry stream")

    if not reasons:
        reasons.append("All readings are within normal operating ranges")

    return reasons

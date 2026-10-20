"""
maintenance.py
---------------
Turns telemetry + health score + anomaly info + failure probability into a
concrete maintenance recommendation (Feature 9). Rule-based and
deterministic on purpose - a maintenance tech should be able to read the
"possible_cause" and immediately understand why the system is
recommending an action.
"""

import config

_PRIORITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _bump(current: str, candidate: str) -> str:
    return candidate if _PRIORITY_ORDER[candidate] > _PRIORITY_ORDER[current] else current


def generate_recommendation(telemetry: dict, health_status: str, failure_risk: str, reasons: list) -> dict:
    """
    telemetry: {"temperature", "rpm", "vibration"}
    health_status: "HEALTHY" | "WARNING" | "CRITICAL"
    failure_risk: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reasons: explainability reasons from health.generate_reasons(), used to
             decide which single action is most relevant right now.

    Returns {"maintenance_priority", "maintenance_action", "possible_cause"}
    """
    temp = telemetry["temperature"]
    rpm = telemetry["rpm"]
    vib = telemetry["vibration"]

    t_low, t_high = config.NORMAL_TEMP_RANGE
    r_low, r_high = config.NORMAL_RPM_RANGE
    v_low, v_high = config.NORMAL_VIBRATION_RANGE

    vib_bad = vib > v_high
    temp_bad = temp > t_high
    rpm_bad = rpm < r_low or rpm > r_high
    issue_count = sum([vib_bad, temp_bad, rpm_bad])

    # --- Priority: worst of health_status and failure_risk ---
    status_to_priority = {"HEALTHY": "LOW", "WARNING": "MEDIUM", "CRITICAL": "CRITICAL"}
    priority = "LOW"
    priority = _bump(priority, status_to_priority.get(health_status, "LOW"))
    priority = _bump(priority, failure_risk)

    # --- Action + cause selection ---
    if issue_count >= 2:
        action = "Schedule preventive maintenance and inspect the bearing/motor assembly."
        cause = "Combined degradation - multiple readings (vibration/temperature/RPM) are simultaneously abnormal."
    elif vib_bad:
        action = "Inspect motor bearings and shaft alignment."
        cause = f"Excessive vibration ({vib:.4f} g, normal is {v_low:.2f}-{v_high:.2f} g)."
    elif temp_bad:
        action = "Check cooling system, lubrication and thermal load."
        cause = f"Elevated temperature ({temp:.1f}\u00b0C, normal is {t_low:.0f}-{t_high:.0f}\u00b0C)."
    elif rpm_bad:
        action = "Inspect motor, drive system and mechanical load."
        cause = f"Unstable RPM ({rpm:.0f}, normal is {r_low:.0f}-{r_high:.0f})."
    elif priority in ("MEDIUM", "HIGH", "CRITICAL"):
        # Nothing crossed an absolute threshold yet but trend/anomaly history
        # is pushing risk up - recommend a precautionary check.
        action = "Schedule a precautionary inspection based on early degradation trend."
        cause = "Readings are still within range, but recent trend and anomaly history indicate developing wear."
    else:
        action = "No action required - continue routine monitoring."
        cause = "All readings within normal operating parameters."

    return {
        "maintenance_priority": priority,
        "maintenance_action": action,
        "possible_cause": cause,
    }

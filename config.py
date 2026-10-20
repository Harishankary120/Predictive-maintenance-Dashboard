"""
config.py
---------
Centralized configuration for the Predictive Maintenance Telemetry Dashboard.

All previously hardcoded values (MQTT host/port, DB credentials, normal
operating ranges, machine list, etc.) live here. Everything can be
overridden with environment variables so the same code works whether you
run it on your host machine (talking to the dockerized Mosquitto/Timescale
containers on localhost) or inside a container on a docker network.

Nothing here is a secret you should commit in a real product - for this
prototype/demo the defaults intentionally match the existing
docker-compose.yml so the project keeps working out of the box.
"""

import os


def _env_float(name, default):
    return float(os.getenv(name, default))


def _env_int(name, default):
    return int(os.getenv(name, default))


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = _env_int("MQTT_PORT", 1883)

# Base topic pattern. {machine_id} is substituted per-machine.
MQTT_TOPIC_TEMPLATE = os.getenv("MQTT_TOPIC_TEMPLATE", "factory/{machine_id}/telemetry")
# Wildcard used by the backend to subscribe to every machine at once.
MQTT_TOPIC_WILDCARD = os.getenv("MQTT_TOPIC_WILDCARD", "factory/+/telemetry")
# Control channel used by the backend to ask the simulator to escalate a
# fault on a specific machine (used by the "Inject Fault" demo button).
MQTT_CONTROL_TOPIC = os.getenv("MQTT_CONTROL_TOPIC", "factory/control/inject_fault")

# ---------------------------------------------------------------------------
# Database (TimescaleDB / PostgreSQL)
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = _env_int("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME", "telemetry_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password123")

# ---------------------------------------------------------------------------
# Machines simulated / tracked by the system
# ---------------------------------------------------------------------------
MACHINE_IDS = os.getenv(
    "MACHINE_IDS",
    "CNC_MACHINE_01,CNC_MACHINE_02,CNC_MACHINE_03,CNC_MACHINE_04",
).split(",")

# ---------------------------------------------------------------------------
# Normal operating ranges (used by health scoring, anomaly rules, the ML
# fallback, and the frontend explainability copy). Keep these consistent
# across simulator.py, health.py, ml_model.py and generate_dataset.py.
# ---------------------------------------------------------------------------
NORMAL_TEMP_RANGE = (_env_float("NORMAL_TEMP_MIN", 60.0), _env_float("NORMAL_TEMP_MAX", 75.0))
NORMAL_RPM_RANGE = (_env_float("NORMAL_RPM_MIN", 1180.0), _env_float("NORMAL_RPM_MAX", 1220.0))
NORMAL_VIBRATION_RANGE = (_env_float("NORMAL_VIB_MIN", 0.02), _env_float("NORMAL_VIB_MAX", 0.08))

# Absolute danger ceilings/floors - values beyond these are always "bad"
# regardless of trend, used for the rule-based fallback + explainability.
DANGER_TEMP = _env_float("DANGER_TEMP", 90.0)
DANGER_VIBRATION = _env_float("DANGER_VIBRATION", 0.18)
DANGER_RPM_LOW = _env_float("DANGER_RPM_LOW", 950.0)
DANGER_RPM_HIGH = _env_float("DANGER_RPM_HIGH", 1300.0)

# ---------------------------------------------------------------------------
# History / trend windows
# ---------------------------------------------------------------------------
HISTORY_WINDOW = _env_int("HISTORY_WINDOW", 30)     # readings kept per machine for trend/rolling stats
TREND_SPLIT = _env_int("TREND_SPLIT", 5)            # readings used per "half" when computing simple trend
ANOMALY_WINDOW = _env_int("ANOMALY_WINDOW", 10)      # readings looked at for "recent anomaly count"
ANOMALY_Z_THRESHOLD = _env_float("ANOMALY_Z_THRESHOLD", 2.5)

# ---------------------------------------------------------------------------
# Model artifact locations
# ---------------------------------------------------------------------------
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/training_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/failure_model.pkl")

# ---------------------------------------------------------------------------
# Health score status thresholds
# ---------------------------------------------------------------------------
HEALTH_HEALTHY_MIN = _env_int("HEALTH_HEALTHY_MIN", 80)
HEALTH_WARNING_MIN = _env_int("HEALTH_WARNING_MIN", 50)

# ---------------------------------------------------------------------------
# Failure risk categories (probability -> label)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = [
    (0.30, "LOW"),
    (0.60, "MEDIUM"),
    (0.80, "HIGH"),
    (1.01, "CRITICAL"),
]


def risk_category(probability: float) -> str:
    for upper_bound, label in RISK_THRESHOLDS:
        if probability < upper_bound:
            return label
    return "CRITICAL"

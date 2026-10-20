"""
main.py
-------
FastAPI backend for the Predictive Maintenance Telemetry Dashboard.

Pipeline (unchanged at the architecture level - just extended):

    simulator.py (N machines)
        -> MQTT (Mosquitto), wildcard topic factory/+/telemetry
        -> this file: anomaly detection, trend/rolling stats, health score,
           ML failure prediction, maintenance recommendation, explainability
        -> TimescaleDB/PostgreSQL (telemetry + alerts)
        -> WebSocket /ws/telemetry (enriched payload)
        -> Vue 3 dashboard (index.html)

New in this version: multi-machine support, machine health scoring,
ML-based failure prediction with explainability, maintenance
recommendations, alert history, and REST endpoints for the dashboard to
pull machine/alert/health/prediction/history data on demand.
"""

import json
import time
import asyncio
from collections import deque, defaultdict

import numpy as np
import psycopg2
import psycopg2.pool
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt

import config
import health
import maintenance
import ml_model

app = FastAPI(title="Predictive Maintenance Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database connection pool (Feature 21: avoid opening a raw connection per
# insert/query - reuse a small pool instead).
# ---------------------------------------------------------------------------
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=config.DB_HOST,
    port=config.DB_PORT,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
    dbname=config.DB_NAME,
)


class db_cursor:
    """Small context manager: borrow a pooled connection, commit/rollback, return it."""

    def __enter__(self):
        self.conn = db_pool.getconn()
        self.cur = self.conn.cursor()
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.cur.close()
            db_pool.putconn(self.conn)


def init_db():
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS machine_telemetry (
                    timestamp DOUBLE PRECISION,
                    machine_id VARCHAR(50),
                    temperature REAL,
                    rpm REAL,
                    vibration REAL,
                    is_anomaly BOOLEAN
                );
                """
            )
            for column, coltype in [
                ("health_score", "INTEGER"),
                ("health_status", "VARCHAR(20)"),
                ("failure_probability", "REAL"),
                ("failure_risk", "VARCHAR(20)"),
                ("maintenance_priority", "VARCHAR(20)"),
                ("machine_status", "VARCHAR(30)"),
            ]:
                cur.execute(
                    f"ALTER TABLE machine_telemetry ADD COLUMN IF NOT EXISTS {column} {coltype};"
                )

            # --- ADD THIS SQL VIEW FOR GRAFANA ---
            cur.execute(
                """
                CREATE OR REPLACE VIEW grafana_telemetry AS
                SELECT 
                    to_timestamp(timestamp) AS time,
                    machine_id,
                    temperature,
                    rpm,
                    vibration,
                    health_score,
                    failure_probability
                FROM machine_telemetry;
                """
            )
        print("✅ DB & Grafana SQL View initialized successfully!")
    except Exception as e:
        print(f"⚠️ DB Init Warning: {e}")


init_db()

# ---------------------------------------------------------------------------
# In-memory per-machine state
# ---------------------------------------------------------------------------
# Rolling raw-reading history per machine, used for trend/rolling stats and
# ML feature engineering. Each item: {temperature, rpm, vibration, timestamp, is_anomaly}
machine_history = defaultdict(lambda: deque(maxlen=config.HISTORY_WINDOW))

# Separate short vibration buffer per machine, purely for the statistical
# (Z-score) anomaly detector - keeps the original detection approach intact.
vibration_history = defaultdict(list)

# Latest enriched state per machine - what /api/machines and the dashboard read.
latest_state = {}

# Recent alerts (most-recent-first), capped in memory; full history in DB.
recent_alerts = deque(maxlen=100)

# last time each machine was heard from, for OFFLINE detection
last_seen = {}

active_connections = set()


def detect_anomaly(machine_id, vibration_value):
    hist = vibration_history[machine_id]
    hist.append(float(vibration_value))
    if len(hist) > 30:
        hist.pop(0)

    if len(hist) < 5:
        return False

    mean = float(np.mean(hist))
    std = float(np.std(hist))
    if std == 0:
        return False

    z_score = (float(vibration_value) - mean) / std
    return bool(abs(z_score) > config.ANOMALY_Z_THRESHOLD)


def save_telemetry_to_db(data, enriched):
    try:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO machine_telemetry
                    (timestamp, machine_id, temperature, rpm, vibration, is_anomaly,
                     health_score, health_status, failure_probability, failure_risk,
                     maintenance_priority, machine_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    float(data["timestamp"]),
                    str(data["machine_id"]),
                    float(data["temperature"]),
                    float(data["rpm"]),
                    float(data["vibration"]),
                    bool(data["is_anomaly_detected"]),
                    int(enriched["health_score"]),
                    str(enriched["health_status"]),
                    float(enriched["failure_probability"]),
                    str(enriched["failure_risk"]),
                    str(enriched["maintenance_priority"]),
                    str(enriched["machine_status"]),
                ),
            )
    except Exception as e:
        print(f"Error saving telemetry to DB: {e}")


def save_alert_to_db(alert):
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO alerts (timestamp, machine_id, alert_type, severity, message) "
                "VALUES (%s, %s, %s, %s, %s)",
                (alert["timestamp"], alert["machine_id"], alert["alert_type"], alert["severity"], alert["message"]),
            )
    except Exception as e:
        print(f"Error saving alert to DB: {e}")


async def broadcast_ws(data):
    for ws in list(active_connections):
        try:
            await ws.send_json(data)
        except Exception:
            active_connections.discard(ws)


loop = asyncio.get_event_loop()


def determine_status(health_status, failure_risk, is_anomaly):
    if health_status == "CRITICAL" or failure_risk == "CRITICAL":
        return "CRITICAL"
    if failure_risk == "HIGH" or (health_status == "WARNING" and is_anomaly):
        return "MAINTENANCE_REQUIRED"
    if health_status == "WARNING" or failure_risk == "MEDIUM":
        return "WARNING"
    return "RUNNING"


def process_reading(data: dict) -> dict:
    """Runs the full enrichment pipeline for one incoming telemetry reading."""
    machine_id = data["machine_id"]

    is_anomaly = detect_anomaly(machine_id, data["vibration"])
    data["is_anomaly_detected"] = bool(is_anomaly)

    telemetry = {
        "temperature": float(data["temperature"]),
        "rpm": float(data["rpm"]),
        "vibration": float(data["vibration"]),
    }

    history_before = list(machine_history[machine_id])  # oldest -> newest, excludes current

    prediction = ml_model.predict_failure(telemetry, history_before)
    features = prediction["features"]
    trends = {
        "temperature": features["temp_trend"],
        "rpm": features["rpm_trend"],
        "vibration": features["vibration_trend"],
    }

    anomaly_count = sum(1 for h in history_before if h.get("is_anomaly")) + (1 if is_anomaly else 0)

    health_result = health.calculate_health_score(telemetry, trends, anomaly_count)
    reasons = health.generate_reasons(telemetry, trends, anomaly_count)
    maint = maintenance.generate_recommendation(
        telemetry, health_result["health_status"], prediction["failure_risk"], reasons
    )
    status = determine_status(health_result["health_status"], prediction["failure_risk"], is_anomaly)

    # Now that features have been computed, append this reading to history.
    machine_history[machine_id].append(
        {
            "temperature": telemetry["temperature"],
            "rpm": telemetry["rpm"],
            "vibration": telemetry["vibration"],
            "timestamp": data["timestamp"],
            "is_anomaly": is_anomaly,
        }
    )
    last_seen[machine_id] = time.time()

    enriched = {
        "machine_id": machine_id,
        "timestamp": data["timestamp"],
        "temperature": telemetry["temperature"],
        "rpm": telemetry["rpm"],
        "vibration": telemetry["vibration"],
        "is_anomaly_detected": is_anomaly,
        "health_score": health_result["health_score"],
        "health_status": health_result["health_status"],
        "failure_probability": prediction["failure_probability"],
        "failure_risk": prediction["failure_risk"],
        "prediction_reasons": reasons,
        "maintenance_priority": maint["maintenance_priority"],
        "maintenance_action": maint["maintenance_action"],
        "possible_cause": maint["possible_cause"],
        "machine_status": status,
    }

    latest_state[machine_id] = enriched

    if is_anomaly:
        print(
            f"🚨 [ANOMALY] {machine_id} | Vibration: {telemetry['vibration']} g | "
            f"Health: {health_result['health_score']} | Risk: {prediction['failure_risk']}"
        )

    # --- Alerting: anomaly OR high/critical failure risk OR critical health ---
    if is_anomaly or prediction["failure_risk"] in ("HIGH", "CRITICAL") or health_result["health_status"] == "CRITICAL":
        if prediction["failure_risk"] == "CRITICAL" or health_result["health_status"] == "CRITICAL":
            severity = "CRITICAL"
        elif prediction["failure_risk"] == "HIGH":
            severity = "HIGH"
        else:
            severity = "WARNING"

        if telemetry["vibration"] > config.NORMAL_VIBRATION_RANGE[1]:
            alert_type = "HIGH VIBRATION"
        elif telemetry["temperature"] > config.NORMAL_TEMP_RANGE[1]:
            alert_type = "HIGH TEMPERATURE"
        elif telemetry["rpm"] < config.NORMAL_RPM_RANGE[0] or telemetry["rpm"] > config.NORMAL_RPM_RANGE[1]:
            alert_type = "ABNORMAL RPM"
        else:
            alert_type = "ELEVATED FAILURE RISK"

        alert = {
            "timestamp": data["timestamp"],
            "machine_id": machine_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": maint["maintenance_action"],
        }
        recent_alerts.appendleft(alert)
        save_alert_to_db(alert)
        enriched["new_alert"] = alert

    save_telemetry_to_db(data, enriched)
    return enriched


def on_mqtt_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        enriched = process_reading(data)
        asyncio.run_coroutine_threadsafe(broadcast_ws(enriched), loop)
    except Exception as e:
        print(f"Error processing message: {e}")


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_mqtt_message


async def offline_checker():
    """Background task: marks a machine OFFLINE if no telemetry has arrived recently."""
    OFFLINE_AFTER_SECONDS = 8
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for machine_id, seen_at in list(last_seen.items()):
            state = latest_state.get(machine_id)
            if state and now - seen_at > OFFLINE_AFTER_SECONDS and state.get("machine_status") != "OFFLINE":
                state["machine_status"] = "OFFLINE"
                await broadcast_ws(state)


@app.on_event("startup")
def startup_event():
    mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
    mqtt_client.subscribe(config.MQTT_TOPIC_WILDCARD)
    mqtt_client.loop_start()
    asyncio.get_event_loop().create_task(offline_checker())
    print(f"✅ FastAPI connected to MQTT Broker, subscribed to '{config.MQTT_TOPIC_WILDCARD}'")


@app.get("/")
def read_root():
    return {"status": "Predictive Telemetry Engine Active", "machines_tracked": list(latest_state.keys())}


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    # Send current snapshot immediately so a newly-connected dashboard isn't empty.
    for state in latest_state.values():
        await websocket.send_json(state)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.discard(websocket)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.get("/api/machines")
def get_machines():
    """Summary for every known machine - powers the dashboard summary row + selector."""
    return list(latest_state.values())


@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str):
    if machine_id not in latest_state:
        return {"error": f"Unknown or not-yet-seen machine_id: {machine_id}"}
    return latest_state[machine_id]


@app.get("/api/alerts")
def get_alerts(limit: int = 20):
    return list(recent_alerts)[:limit]


@app.get("/api/health/{machine_id}")
def get_health(machine_id: str):
    if machine_id not in latest_state:
        return {"error": f"Unknown or not-yet-seen machine_id: {machine_id}"}
    state = latest_state[machine_id]
    return {
        "machine_id": machine_id,
        "health_score": state["health_score"],
        "health_status": state["health_status"],
    }


@app.get("/api/prediction/{machine_id}")
def get_prediction(machine_id: str):
    if machine_id not in latest_state:
        return {"error": f"Unknown or not-yet-seen machine_id: {machine_id}"}
    state = latest_state[machine_id]
    return {
        "machine_id": machine_id,
        "failure_probability": state["failure_probability"],
        "failure_risk": state["failure_risk"],
        "prediction_reasons": state["prediction_reasons"],
    }


@app.get("/api/history/{machine_id}")
def get_history(machine_id: str):
    """Recent raw readings for this machine - used to seed the dashboard charts."""
    return list(machine_history.get(machine_id, []))


@app.post("/api/inject-fault")
def inject_fault(machine_id: str = None):
    """
    Manual fault injection endpoint for live demonstrations (Feature 23/24).

    Rather than faking a single telemetry spike, this publishes a control
    message that tells simulator.py to escalate the target machine through
    a realistic degradation sequence (rising vibration/temperature, RPM
    instability, then a critical state) over the following readings - so
    the health score, failure probability, risk, and maintenance
    recommendation all visibly move in real time on the dashboard.
    """
    target = machine_id or config.MACHINE_IDS[0]
    payload = {"machine_id": target, "triggered_at": time.time()}
    mqtt_client.publish(config.MQTT_CONTROL_TOPIC, json.dumps(payload))
    return {"status": "Fault injection triggered", "machine_id": target}

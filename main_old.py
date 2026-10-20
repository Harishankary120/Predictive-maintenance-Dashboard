import json
import time
import asyncio
import numpy as np
import psycopg2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt

app = FastAPI(title="Predictive Maintenance Engine")

# Enable CORS for Vue.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to TimescaleDB
def get_db():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="admin",
        password="password123",
        dbname="telemetry_db"
    )

# Create database table if not exists
def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS machine_telemetry (
                timestamp DOUBLE PRECISION,
                machine_id VARCHAR(50),
                temperature REAL,
                rpm REAL,
                vibration REAL,
                is_anomaly BOOLEAN
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ TimescaleDB table initialized successfully!")
    except Exception as e:
        print(f"⚠️ DB Init Warning: {e}")

init_db()

# Rolling buffer for statistical anomaly detection (last 30 readings)
vibration_history = []
active_connections = set()

def detect_anomaly(vibration_value):
    vibration_history.append(float(vibration_value))
    if len(vibration_history) > 30:
        vibration_history.pop(0)

    if len(vibration_history) < 5:
        return False # Need at least 5 readings to compute baseline

    mean = float(np.mean(vibration_history))
    std = float(np.std(vibration_history))

    if std == 0:
        return False

    # Calculate Z-score (standard deviations away from average)
    z_score = (float(vibration_value) - mean) / std
    
    # Strictly return a standard Python bool
    return bool(abs(z_score) > 2.5)

def save_to_db(data, is_anomaly):
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Explicit Python type casting for psycopg2 adapter compatibility
        ts = float(data['timestamp'])
        m_id = str(data['machine_id'])
        temp = float(data['temperature'])
        rpm = float(data['rpm'])
        vib = float(data['vibration'])
        anomaly = bool(is_anomaly)

        cur.execute(
            "INSERT INTO machine_telemetry (timestamp, machine_id, temperature, rpm, vibration, is_anomaly) VALUES (%s, %s, %s, %s, %s, %s)",
            (ts, m_id, temp, rpm, vib, anomaly)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving to DB: {e}")

async def broadcast_ws(data):
    for ws in list(active_connections):
        try:
            await ws.send_json(data)
        except Exception:
            active_connections.remove(ws)

loop = asyncio.get_event_loop()

def on_mqtt_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        
        is_anomaly = detect_anomaly(data['vibration'])
        data['is_anomaly_detected'] = bool(is_anomaly)
        
        if is_anomaly:
            print(f"🚨 [CRITICAL ANOMALY] Vibration Spike Detected! Machine: {data['machine_id']} | Vibration: {data['vibration']} g")
        
        save_to_db(data, is_anomaly)
        
        # Stream live data to connected WebSockets
        asyncio.run_coroutine_threadsafe(broadcast_ws(data), loop)
    except Exception as e:
        print(f"Error processing message: {e}")

# MQTT Subscriber Setup
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_mqtt_message

@app.on_event("startup")
def startup_event():
    mqtt_client.connect("localhost", 1883, 60)
    mqtt_client.subscribe("factory/machine_01/telemetry")
    mqtt_client.loop_start()
    print("✅ FastAPI connected to MQTT Broker & listening for stream!")

@app.get("/")
def read_root():
    return {"status": "Predictive Telemetry Engine Active"}

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# Manual Fault Injection Endpoint (Outside of WebSocket handler)
@app.post("/api/inject-fault")
def inject_fault():
    """Manual fault injection endpoint for live judge demonstrations."""
    fault_payload = {
        "timestamp": time.time(),
        "machine_id": "CNC_MACHINE_01",
        "temperature": 82.5,
        "rpm": 1105.0,
        "vibration": 0.3450  # High vibration spike breaching Z-score
    }
    mqtt_client.publish("factory/machine_01/telemetry", json.dumps(fault_payload))
    return {"status": "Fault injected successfully"}
"""
simulator.py
-------------
Factory telemetry simulator (Feature 6 + Feature 7).

Simulates several CNC machines publishing over MQTT, each with its own
independent condition state machine instead of pure random spikes:

    NORMAL -> (spontaneously, rarely) one of:
        OVERHEATING, BEARING_WEAR, IMBALANCE, MOTOR_DEGRADATION
    -> degradation_level climbs gradually each second (not an instant jump)
    -> readings drift from normal toward the condition's worst-case values
    -> eventually held at a critical level, then "repaired" back to NORMAL

It also listens on an MQTT control topic
(config.MQTT_CONTROL_TOPIC = "factory/control/inject_fault"). When
main.py's "Inject Fault (Demo)" button publishes a message there, the
target machine is pushed into an accelerated degradation sequence so the
dashboard visibly moves from HEALTHY -> WARNING -> CRITICAL over the
following several seconds, rather than a single fake spike.

Keeps the original MQTT architecture: still publishes plain JSON telemetry
to factory/{machine_id}/telemetry once per second.
"""

import json
import random
import time
import paho.mqtt.client as mqtt

import config

CONDITIONS = ["OVERHEATING", "BEARING_WEAR", "IMBALANCE", "MOTOR_DEGRADATION"]

# Chance, per machine per second, that a healthy machine spontaneously
# starts degrading - kept low so most of a demo run looks healthy.
SPONTANEOUS_FAULT_PROB = 0.01
# Chance, per second, that a machine sitting at full (>=0.95) degradation
# gets "repaired" back to NORMAL, so the demo naturally cycles over time.
RECOVERY_PROB = 0.03


class MachineState:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.topic = config.MQTT_TOPIC_TEMPLATE.format(machine_id=machine_id)
        self.condition = "NORMAL"
        self.degradation_level = 0.0
        self.accelerated = False

    def maybe_start_fault(self, condition=None, accelerated=False):
        self.condition = condition or random.choice(CONDITIONS)
        self.degradation_level = max(self.degradation_level, 0.15 if not accelerated else 0.55)
        self.accelerated = accelerated

    def step(self):
        if self.condition == "NORMAL":
            if random.random() < SPONTANEOUS_FAULT_PROB:
                self.maybe_start_fault()
            return

        increment = random.uniform(0.06, 0.15) if self.accelerated else random.uniform(0.01, 0.04)
        self.degradation_level = min(1.0, self.degradation_level + increment)

        if self.degradation_level >= 0.98 and random.random() < RECOVERY_PROB:
            # Simulated repair/maintenance intervention.
            self.condition = "NORMAL"
            self.degradation_level = 0.0
            self.accelerated = False

    def generate_telemetry(self):
        d = self.degradation_level

        # Baseline healthy values (matches config.NORMAL_* ranges).
        temp = random.uniform(63, 72)
        rpm = random.uniform(1190, 1210)
        vib = random.uniform(0.03, 0.06)

        if self.condition == "OVERHEATING":
            temp += d * random.uniform(25, 35)
            rpm += random.uniform(-15, 15) * d
            vib += d * random.uniform(0.01, 0.03)
        elif self.condition == "BEARING_WEAR":
            vib += d * random.uniform(0.15, 0.28)
            temp += d * random.uniform(5, 12)
            rpm += random.uniform(-10, 10) * d
        elif self.condition == "IMBALANCE":
            rpm += random.choice([-1, 1]) * d * random.uniform(150, 260)
            vib += d * random.uniform(0.06, 0.14)
            temp += d * random.uniform(3, 8)
        elif self.condition == "MOTOR_DEGRADATION":
            # Worst case: everything degrades together.
            temp += d * random.uniform(20, 32)
            rpm -= d * random.uniform(200, 300)
            vib += d * random.uniform(0.14, 0.26)

        temp = round(max(20.0, min(160.0, temp)), 2)
        rpm = round(max(0.0, min(2200.0, rpm)), 2)
        vib = round(max(0.0, min(1.0, vib)), 4)

        return {
            "machine_id": self.machine_id,
            "timestamp": time.time(),
            "temperature": temp,
            "rpm": rpm,
            "vibration": vib,
            "is_simulated_fault": self.condition != "NORMAL",
            "condition": self.condition,
        }


machines = {mid: MachineState(mid) for mid in config.MACHINE_IDS}


def on_control_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        target = payload.get("machine_id")
        state = machines.get(target)
        if state:
            state.maybe_start_fault(condition=random.choice(CONDITIONS), accelerated=True)
            print(f"⚡ [FAULT INJECTED] {target} escalating toward failure...")
        else:
            print(f"⚠️ Fault injection requested for unknown machine_id: {target}")
    except Exception as e:
        print(f"Error handling control message: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.message_callback_add(config.MQTT_CONTROL_TOPIC, on_control_message)

try:
    client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
    client.subscribe(config.MQTT_CONTROL_TOPIC)
    client.loop_start()
    print("✅ Connected to MQTT Broker successfully!")
except Exception as e:
    print(f"❌ Could not connect to MQTT Broker: {e}")
    exit(1)

print(f"🏭 Simulating {len(machines)} machines: {', '.join(machines.keys())}")
print("Press Ctrl + C to stop.")

try:
    while True:
        for state in machines.values():
            state.step()
            data = state.generate_telemetry()
            client.publish(state.topic, json.dumps(data))

            flag = "⚠️ [FAULT]" if data["is_simulated_fault"] else "🟢 [NORMAL]"
            print(
                f"📡 {flag} {data['machine_id']:16s} ({state.condition:16s}) "
                f"Temp: {data['temperature']:6.1f}°C | RPM: {data['rpm']:7.1f} | Vibration: {data['vibration']:.4f} g"
            )
        time.sleep(1)  # one telemetry round per second, matching the original cadence
except KeyboardInterrupt:
    print("\n🛑 Simulator stopped.")
    client.loop_stop()
    client.disconnect()

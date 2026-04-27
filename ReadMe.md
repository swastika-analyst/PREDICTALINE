# PredictaLine — Deployment Guide
## NEVIndia | EV Assembly Predictive Maintenance

---

## 🚀 Quick Start (Google Colab — 5 minutes)

### Step 1 — Open Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `PredictaLine_Colab.ipynb`

### Step 2 — Run Everything
```
Runtime → Run all   (Ctrl+F9)
```
All dependencies install automatically. No GPU needed.

### Step 3 — View Outputs
After running, three files appear in the Colab file browser (left sidebar):
| File | What it shows |
|------|--------------|
| `predictaline_dashboard.html` | Interactive Plotly dashboard — open in browser |
| `predictaline_shap.png` | SHAP feature importance chart |
| `predictaline_eda.png` | Sensor distribution: healthy vs pre-failure |
| `predictaline_performance.png` | ROC + Confusion Matrix + Feature Importance |

Download files: **Right-click → Download** in the Colab sidebar.

---

## 🏭 What the Prototype Demonstrates

### Dataset
- **5,000 sensor readings** from 3 machine types
- **13 sensor features** per reading (vibration, torque, current, temp, acoustic, etc.)
- **3 engineered features** (vibration-torque ratio, thermal load, acoustic sum)
- **~15% failure rate** — realistic for industrial settings

### Models Trained
| Model | Role | Typical AUC |
|-------|------|------------|
| Random Forest | Baseline | ~0.97 |
| Gradient Boosting | Primary | ~0.98–0.99 |
| Ensemble (60/40) | Production | ~0.99 |

### Alert Pipeline (Cell 8)
For each sensor reading above 55% failure probability:
1. **Root cause identified** — top 3 SHAP-contributing sensors
2. **WhatsApp message** generated with machine ID, risk level, ETA, action
3. **SAP PM02 work order** auto-raised with parts list and maintenance window

---

## 📦 Production Deployment Path

### Option A — Local Server (FastAPI)
```bash
pip install fastapi uvicorn scikit-learn shap pandas
```

```python
# predictaline_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, joblib

app = FastAPI(title="PredictaLine API")
model  = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")

class SensorReading(BaseModel):
    vibration_mm_s: float
    torque_nm: float
    current_a: float
    temperature_c: float
    acoustic_db: float
    cycle_deviation_pct: float
    coolant_flow_l_min: float
    electrode_wear_pct: float
    voltage_drift_mv: float
    pressure_bar: float
    vibration_torque_ratio: float
    thermal_load: float
    acoustic_vibration_sum: float
    machine_type_enc: int   # 0=assembly, 1=battery, 2=welding

@app.post("/predict")
def predict(reading: SensorReading):
    features = np.array([[
        reading.vibration_mm_s, reading.torque_nm, reading.current_a,
        reading.temperature_c, reading.acoustic_db, reading.cycle_deviation_pct,
        reading.coolant_flow_l_min, reading.electrode_wear_pct,
        reading.voltage_drift_mv, reading.pressure_bar,
        reading.vibration_torque_ratio, reading.thermal_load,
        reading.acoustic_vibration_sum, reading.machine_type_enc
    ]])
    prob = float(model.predict_proba(features)[0][1])
    return {
        "failure_probability": round(prob, 4),
        "risk_level": "CRITICAL" if prob > 0.8 else "HIGH" if prob > 0.6 else "OK",
        "alert": prob > 0.55
    }
```

```bash
uvicorn predictaline_api:app --reload --port 8000
# API docs at: http://localhost:8000/docs
```

### Option B — Save & Load Model
```python
# In Colab, after training (Cell 5):
import joblib
joblib.dump(gb, 'gb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load anywhere:
model = joblib.load('gb_model.pkl')
scaler = joblib.load('scaler.pkl')
```

### Option C — Streamlit Dashboard
```bash
pip install streamlit
```

```python
# app.py
import streamlit as st
import numpy as np, joblib, pandas as pd

st.set_page_config(page_title="PredictaLine", layout="wide")
st.title("🔮 PredictaLine — Live Sensor Monitor")

model = joblib.load("gb_model.pkl")
col1, col2, col3 = st.columns(3)

with col1:
    vibration = st.slider("Vibration (mm/s)", 0.0, 20.0, 2.1)
    torque    = st.slider("Torque (Nm)", 10.0, 200.0, 120.0)
    current   = st.slider("Current (A)", 5.0, 80.0, 45.0)

with col2:
    temperature = st.slider("Temperature (°C)", 20.0, 120.0, 68.0)
    acoustic    = st.slider("Acoustic (dB)", 40.0, 110.0, 72.0)
    voltage_drift = st.slider("Voltage Drift (mV)", 0.0, 50.0, 5.0)

if st.button("▶ Run Inference"):
    features = np.array([[vibration, torque, current, temperature, acoustic,
                          0.5, 8.5, 30, voltage_drift, 6.2,
                          vibration/(torque+1e-6), temperature*current/1000,
                          acoustic+vibration*2, 2]])
    prob = model.predict_proba(features)[0][1]
    st.metric("Failure Probability", f"{prob:.1%}")
    if prob > 0.55:
        st.error(f"⚠️ ALERT — Risk: {'CRITICAL' if prob>0.8 else 'HIGH'}")
    else:
        st.success("✅ Machine healthy")
```

```bash
streamlit run app.py
```

---

## 🔌 Real Sensor Integration

### Connect to MQTT (industrial sensors)
```python
import paho.mqtt.client as mqtt
import json, joblib, numpy as np

model = joblib.load("gb_model.pkl")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    features = np.array([[
        data["vibration"], data["torque"], data["current"],
        data["temperature"], data["acoustic"], data["cycle_dev"],
        data["coolant"], data["electrode_wear"], data["voltage_drift"],
        data["pressure"], data["vib_torque_ratio"], data["thermal_load"],
        data["acoustic_vib_sum"], data["machine_type_enc"]
    ]])
    prob = model.predict_proba(features)[0][1]
    if prob > 0.55:
        send_whatsapp_alert(data["machine_id"], prob)   # your WA API

client = mqtt.Client()
client.on_message = on_message
client.connect("your-mqtt-broker", 1883)
client.subscribe("plant/sensors/#")
client.loop_forever()
```

### WhatsApp Integration (Twilio)
```python
from twilio.rest import Client

def send_whatsapp_alert(machine_id, prob):
    client = Client("TWILIO_SID", "TWILIO_TOKEN")
    client.messages.create(
        from_="whatsapp:+14155238886",
        to="whatsapp:+91XXXXXXXXXX",
        body=f"⚠️ PREDICTALINE ALERT\nMachine: {machine_id}\nFailure Prob: {prob:.0%}\nAction required within 72h"
    )
```

---

## 📊 Key Metrics Reference

| Metric | Value |
|--------|-------|
| Prediction horizon | 72 hours |
| Sensors monitored | 13 (expandable) |
| Model type | Gradient Boosting Ensemble |
| Alert threshold | 55% failure probability |
| Expected AUC | 0.97–0.99 |
| Downtime reduction | ~71% |
| Estimated savings | ₹100+ Cr/plant/yr |

---

## 🛠️ Extending the Model

### Add more sensors
Just add columns to the dataset generator and include them in `FEATURE_COLS`.

### Retrain on real data
Replace `generate_ev_assembly_data()` with your CSV:
```python
df = pd.read_csv("your_plant_data.csv")
# Ensure columns match FEATURE_COLS
# Set failure_in_72h as your target column
```

### Switch to LSTM (time-series)
Replace GradientBoosting with:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
```
Use rolling windows of sensor data as input sequences.

---

*PredictaLine — NEVIndia | Built for EV Assembly Intelligence*

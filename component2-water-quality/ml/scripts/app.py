import serial
import time
import numpy as np
import pandas as pd
import joblib
import gradio as gr

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("../models/water_quality_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

print("✅ Model Loaded Successfully")

# ==========================================================
# CONNECT TO ARDUINO
# ==========================================================

arduino = serial.Serial("/dev/cu.usbmodem21401", 9600, timeout=2)

time.sleep(3)

# Clear startup messages
arduino.reset_input_buffer()

print("✅ Arduino Connected")

# ==========================================================
# READ SENSOR VALUES
# ==========================================================

def read_sensor():

    while True:

        line = arduino.readline().decode(errors="ignore").strip()

        if line == "":
            continue

        print("Received:", line)

        if line == "System Ready":
            continue

        values = line.split(",")

        if len(values) != 5:
            continue

        try:

            ph = float(values[0])
            temperature = float(values[1])
            chlorine = float(values[2])
            turbidity = float(values[3])
            tds = float(values[4])

            return ph, temperature, chlorine, turbidity, tds

        except ValueError:
            continue


# ==========================================================
# PREDICTION
# ==========================================================

def predict():

    ph, temperature, chlorine, turbidity, tds = read_sensor()

    sample = pd.DataFrame(
        [[ph, temperature, chlorine, turbidity, tds]],
        columns=[
            "pH",
            "Temperature",
            "Chlorine",
            "Turbidity",
            "TDS"
        ]
    )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    status = label_encoder.inverse_transform(prediction)[0]

    if status == "SAFE":
        result = "🟢 SAFE\n\nWater quality is suitable."

    elif status == "WARNING":
        result = "🟡 WARNING\n\nWater quality needs attention."

    else:
        result = "🔴 CRITICAL\n\nWater quality is unsafe."

    return (
        ph,
        temperature,
        chlorine,
        turbidity,
        tds,
        result
    )


# ==========================================================
# UI
# ==========================================================

with gr.Blocks(
    title="AI Water Quality Monitoring",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 🌊 AI Water Quality Monitoring System

### Arduino + Machine Learning + Gradio Dashboard
"""
    )

    with gr.Row():

        ph_box = gr.Number(
            label="🧪 pH",
            interactive=False
        )

        temp_box = gr.Number(
            label="🌡 Temperature (°C)",
            interactive=False
        )

        chlorine_box = gr.Number(
            label="🧴 Chlorine (ppm)",
            interactive=False
        )

    with gr.Row():

        turbidity_box = gr.Number(
            label="💧 Turbidity (NTU)",
            interactive=False
        )

        tds_box = gr.Number(
            label="🧂 TDS (ppm)",
            interactive=False
        )

    prediction_box = gr.Textbox(
        label="🤖 AI Prediction",
        lines=4,
        interactive=False
    )

    check_button = gr.Button(
        "🔄 Check Water Quality",
        variant="primary"
    )

    check_button.click(
        fn=predict,
        outputs=[
            ph_box,
            temp_box,
            chlorine_box,
            turbidity_box,
            tds_box,
            prediction_box
        ]
    )

    gr.Markdown(
        """
---
Developed using **Arduino • Machine Learning • Gradio**
"""
    )

demo.launch()
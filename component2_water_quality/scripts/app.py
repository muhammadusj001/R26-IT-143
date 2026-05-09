import gradio as gr
import numpy as np
import joblib

# ============================================================
# LOAD TRAINED FILES
# ============================================================

model = joblib.load("water_quality_model.pkl")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")

print("✅ Model Loaded Successfully")

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_water_quality(
    ph,
    temperature,
    chlorine,
    turbidity,
    tds
):

    # Create input array
    sample = np.array([
        [ph, temperature, chlorine, turbidity, tds]
    ])

    # Scale values
    sample_scaled = scaler.transform(sample)

    # Predict
    prediction = model.predict(sample_scaled)

    # Decode prediction
    status = label_encoder.inverse_transform(prediction)

    result = status[0]

    # Extra message
    if result == "SAFE":
        message = "✅ Water Quality is SAFE"

    elif result == "WARNING":
        message = "⚠️ Water Quality is WARNING"

    else:
        message = "🚨 Water Quality is CRITICAL"

    return message

# ============================================================
# GRADIO UI
# ============================================================

interface = gr.Interface(
    fn=predict_water_quality,

    inputs=[

        gr.Number(
            label="pH Value",
            value=7.5
        ),

        gr.Number(
            label="Temperature (°C)",
            value=28
        ),

        gr.Number(
            label="Chlorine (ppm)",
            value=2.0
        ),

        gr.Number(
            label="Turbidity (NTU)",
            value=0.8
        ),

        gr.Number(
            label="TDS (ppm)",
            value=400
        )

    ],

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="🌊 Pool Water Quality Prediction System",

    description="""
Enter pool water sensor values to predict water quality status using Machine Learning.
""",

    theme="soft"
)

# ============================================================
# RUN APP
# ============================================================

interface.launch()
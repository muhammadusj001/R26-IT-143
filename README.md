# WaterQ-RP: AI-Based Smart Swimming Pool Monitoring System

An intelligent system for monitoring and ensuring swimming pool water quality using machine learning and real-time analysis.

## Project Overview

This repository contains the implementation of an AI-Based Smart Swimming Pool Monitoring System developed as a SLIIT Final Year Project (R26-IT-143). The system leverages machine learning to predict and monitor pool water quality parameters, providing real-time insights and recommendations for pool maintenance.

## Features

- **Water Quality Prediction**: ML-based prediction of pool water quality metrics
- **Real-time Monitoring**: Monitor water parameters in real-time
- **Web Interface**: Interactive Gradio/Flask web application for easy access
- **Data Analysis**: Comprehensive data analysis and visualization capabilities
- **Dataset**: Augmented dataset with comprehensive water quality metrics

## Project Structure

```
WaterQ-RP/
├── app.py                                    # Main Flask/Gradio application
├── poolwatersafty-rp-final.ipynb            # Jupyter notebook with analysis and modeling
├── pool_water_quality_augmented_dataset.csv # Dataset with water quality metrics
├── water_quality_model.pkl                  # Trained ML model
├── label_encoder.pkl                        # Label encoder for categorical features
├── scaler.pkl                               # Feature scaler for normalization
├── CODE/
│   └── R26-IT-143/                          # Main project directory
│       ├── README.md                        # Project documentation
│       ├── component2_water_quality/
│       │   ├── datasets/
│       │   │   └── pool_water_quality_augmented_dataset.csv
│       │   ├── models/                      # Trained models storage
│       │   └── scripts/
│       │       ├── app.py                   # Application source
│       │       ├── poolwatersafty-rp-final.ipynb
│       │       └── sketch_may9a/
│       │           └── sketch_may9a.ino     # Arduino sketch for hardware integration
└── .gradio/                                 # Gradio configuration

```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- gradio or Flask
- jupyter

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd WaterQ-RP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The web interface will be available at `http://localhost:7860` (for Gradio) or `http://localhost:5000` (for Flask).

## Usage

### Using the Web Interface
1. Start the application using `python app.py`
2. Access the interface through your browser
3. Input water quality parameters
4. Receive predictions and recommendations

### Using the Jupyter Notebook
Open `poolwatersafty-rp-final.ipynb` in Jupyter to:
- Explore the dataset
- Review the model training process
- Analyze water quality metrics
- Generate visualizations

## Dataset

The project uses an augmented dataset containing water quality metrics:
- **File**: `pool_water_quality_augmented_dataset.csv`
- **Features**: Various water quality parameters (pH, chlorine levels, temperature, etc.)

## Models

Pre-trained models are included:
- `water_quality_model.pkl` - RandomForestClassifier model trained on water quality data
- `label_encoder.pkl` - Categorical feature encoding for status classes (SAFE, WARNING, CRITICAL)
- `scaler.pkl` - StandardScaler for feature normalization

### Model Details

**Algorithm**: Random Forest Classifier
- **Purpose**: Classifies pool water quality into three categories: SAFE, WARNING, or CRITICAL
- **Features Used**: pH, Temperature, Chlorine, Turbidity, TDS
- **Training**: Trained on augmented water quality dataset with preprocessing and feature scaling

## Hardware Integration

An Arduino sketch (`sketch_may9a.ino`) is available for hardware sensor integration, allowing real-time data collection from physical water quality sensors.

## Contributors

- SLIIT Final Year Project Team

## License

This project is part of the SLIIT curriculum and is subject to institutional policies.

## Contact & Support

For questions or issues, please refer to the project documentation in the CODE/R26-IT-143 directory.

---

**Last Updated**: May 2026

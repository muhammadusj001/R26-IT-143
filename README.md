# R26-IT-143 AI-Based Smart Swimming Pool Monitoring and Maintenance System

SLIIT Final Year Project — an integrated AI-powered smart pool monitoring platform for safety, maintenance, and real-time decision support.

## Project overview

This system combines four AI modules into one unified dashboard:

- Crowd-aware maintenance scheduling
- Water quality prediction
- Drowning detection
- Garbage intrusion detection

The platform uses a Flask-SocketIO backend and a Next.js frontend, with a lightweight vanilla HTML fallback for simple demos and offline presentation environments.

## Current architecture

- Component 1: real swimmer detection, bather-load estimation, and maintenance scheduling logic
- Component 2: AI-based water quality prediction using sensor readings and trained models
- Component 3: drowning behaviour detection with alert logic for safety monitoring
- Component 4: garbage detection with false-positive filtering for pool objects

The shared dashboard coordinates these modules into a single monitoring experience.

## Branch guide

- main: stable project overview and top-level documentation
- development: integrated system with the full unified dashboard
- component1-crowd-maintenance: Component 1 work
- component2-water-quality: Component 2 work
- component3-drowning-detection: Component 3 work
- component4-garbage-detection: Component 4 work

Each component branch holds that member's focused work, while development contains the merged integrated system.

## Setup

From the project root:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cd smart-pool-dashboard/backend
python app.py
```

Then open:

```text
http://localhost:5000
```

## Hardware and simulation mode

The system supports both simulation mode and live hardware mode:

- Set CAMERA_SOURCE to switch between simulated camera input and a real camera/device
- Set SENSOR_SOURCE to switch between simulated sensor values and a real hardware sensor stream

This allows the dashboard to run even when physical hardware is unavailable.

## Airline Schedule Optimizer

An AI-powered airline operations management dashboard designed to optimize flight schedules at busy airports like Mumbai (BOM) and Delhi (DEL). This system helps operations teams identify optimal time slots, predict delays, and minimize cascading impacts through intelligent scheduling.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.25+-red.svg)

##  Problem Statement

Due to capacity limitations and heavy passenger load, flight operations at busy airports are becoming a scheduling nightmare. Controllers and operators need to find efficiency in scheduling within system constraints and find means to de-congest flight traffic.

## Features

- **Intelligent Data Transformer**: Auto-normalizes uploaded Excel/CSV files into standardized format
- **AI-Powered Analytics**: Real-time insights using Google Gemini AI integration
- **Delay Prediction**: Machine learning models for predicting flight delays
- **Schedule Optimization**: Recommends optimal time slots to reduce congestion
- **Cascade Impact Analysis**: Identifies flights with highest cascading delay impact
- **Runway Utilization**: Optimizes runway capacity and usage patterns
- **NLP Query Interface**: Natural language processing for operational queries
- **Interactive Dashboard**: Comprehensive Streamlit-based visualization

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows / macOS / Linux
- **Package Manager**: pip
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)

## Project Structure

```
Airline Schedule Optimizer
├── app/
│   ├── main.py                 # Main Streamlit application
│   └── main_updated.py         # Updated version with enhancements
├── src/
│   ├── data_processor.py       # Data processing and transformation
│   ├── optimizer.py           # Schedule optimization algorithms
│   ├── predictor.py           # ML models for delay prediction
│   ├── anomaly_detector.py    # Anomaly detection in flight patterns
│   ├── peak_time_analyzer.py  # Peak time analysis
│   ├── cascade_delay_predictor.py  # Cascade delay analysis
│   ├── nlp_query_processor.py # Natural language query processing
│   └── advanced_optimizer.py  # Advanced optimization algorithms
├── data/
│   ├── flight_schedule_data.csv    # Sample flight data
│   ├── optimized_schedule.csv      # Optimized schedule output
│   └── Flight_Data.xlsx           # Sample Excel data
├── notebooks/
│   └── flight_analysis.ipynb     # Jupyter notebook for analysis
├── docs/
│   ├── flight_radar_integration.md
│   └── openai_setup.md
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                   # This file


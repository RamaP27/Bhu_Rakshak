
<img width="3508" height="2480" alt="bhurak-shak-Symbol" src="https://github.com/user-attachments/assets/0cbada87-e30e-4cf0-91f8-b14f9b97d77d" />

<h1 style="color:olive;">Bhu-Rakshak: A Smart Irrigation Solution</h1>

The **"Bhu-Rakshak"** project proposes a pioneering smart irrigation framework that utilizes the Internet of Things (IoT), Artificial Intelligence (AI), and autonomous drone technology to revolutionize agricultural water management.  

The system aims to create a smart, autonomous, and data-driven ecosystem to ensure optimal resource utilization, enhance crop yield, and promote environmental sustainability. It addresses the challenge of inefficient water management in agriculture, which contributes to water scarcity, soil degradation, and suboptimal crop yields.  

Bhu-Rakshak collects real-time data from in-field soil sensors and meteorological sources, processes it through advanced predictive machine learning models, and executes precise, automated irrigation using a fleet of intelligent drones. This approach represents a paradigm shift towards precision agriculture, promising to significantly reduce water consumption, boost agricultural productivity, and enhance the resilience of food supply chains. 

<h1 style="color:olive;">Opportunities & Differentiation</h1>
Opportunities & Differentiation

Microservices
	Farmers: Increased profitability through higher yields and lower resource costs.
	Governments: Tool for water conservation, drought mitigation, and food security.
	Tech Partners: Groundbreaking AI/IoT application, opening new markets.

Differentiation
	Integrated AI: Predictive model augmented by LLM with RAG for context-aware decisions.
	Autonomous Drones: Precise data analytics with targeted water delivery.
	Microservices: Built for resilience, scalability, and future integration.
  Problem Solving & USP


Problem Solving
	Reduce Water Waste: Cuts down the estimated 50% water wastage in traditional irrigation.
	Boost Productivity: Optimal resource utilization and targeted irrigation lead to healthier crops and significantly higher yields.
	Enhance Food Security: Strengthens the resilience of food supply chains against climate change and resource depletion.

USP
	Intelligent Precision Agriculture: A unique blend of granular IoT data, advanced AI (ML + RAG-LLM), and autonomous drone technology for highly accurate and adaptive irrigation.
	Sustainable & Economically Viable: Delivers significant environmental benefits (water conservation) alongside tangible economic returns for farmers.
	Scalable and Future-Ready Platform: Designed as an "operating system" for precision agriculture, capable of integrating future innovations like pest and nutrient management.

### Solution Features: Data Acquisition & Analytics

Intelligent Data Acquisition:
	Continuous monitoring of soil moisture levels using in-field capacitive and resistive sensors. 	Aggregation of real-time environmental data (temperature, humidity, precipitation).
	Integration of crop-specific information (type, growth stage, water requirements).

	Efficient data transmission over long distances using LoRaWAN protocol.
Advanced Predictive Analytics & Decision-Making:
	Sophisticated Machine Learning Engine utilizing Neural Network architectures like Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU) to forecast crop water needs.
	Integration of Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) framework for enhanced decision-making and access to a vast knowledge base of agricultural research and best practices.
	Dynamic Irrigation Scheduling algorithm to generate optimized irrigation plans. 	Optimized Drone Flight Path generation based on AI predictions.


# Bhu-Rakshak: A Smart Irrigation Solution

## Overview
Bhu-Rakshak is a smart irrigation framework developed for precision agriculture. It combines **IoT sensors**, **machine learning (LSTM-based models)**, and **autonomous decision-making** to optimize water usage in farming. The project addresses the challenge of water scarcity, soil degradation, and suboptimal crop yields by enabling data-driven irrigation control.

This repository contains the code, model pipeline, and experimentation workflow used in the hackathon prototype.

---

## Features
- **Soil & Climate Data Integration**  
  Utilizes soil moisture sensors and meteorological data for real-time monitoring.

- **Machine Learning Model (LSTM)**  
  Predicts irrigation needs based on historical soil moisture and temperature data.

- **Data Preprocessing & Scaling**  
  Normalization, label encoding, and feature engineering for reliable model performance.

- **Visualization Tools**  
  Scatter plots and charts for soil moisture vs. temperature analysis with crop/irrigation status.

- **Gradio Interface**  
  Simple interactive UI to test predictions of irrigation requirements.

- **Cloud & Deployment Ready**  
  Designed for execution in Google Colab with GPU acceleration and Drive integration.

---

## Tech Stack
- **Python**
- **TensorFlow / Keras** (LSTM model)
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn** (data visualization)
- **Gradio** (web interface)
- **Google Colab** (training environment)

---

## Repository Structure
- `Water_Spklr3.ipynb` — Main notebook containing:
  - Data preprocessing
  - Model definition & training
  - Visualization functions
  - Evaluation metrics
  - Gradio-based demo interface

- `sprinkler_lstm_model.pkl` — Serialized trained model (to be saved/loaded for predictions).

---

## Getting Started

### Prerequisites
- Python 3.8+
- Google Colab / Jupyter Notebook
- Required libraries:
  ```bash
  pip install tensorflow scikit-learn pandas numpy matplotlib seaborn gradio joblib


Bhu_Rakshak/
│
├── app.py                 # Main entry point with Gradio UI
├── model.py               # Model loading & prediction
├── preprocess.py          # Data preprocessing helpers
├── requirements.txt       # Dependencies
└── README.md              # Documentation

### model.py

import joblib
import numpy as np
import tensorflow as tf

MODEL_PATH = "sprinkler_lstm_model.pkl"

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def predict_irrigation(model, features: np.ndarray):
    """
    features: numpy array shaped (1, n_features)
    returns: irrigation status (string)
    """
    prediction = model.predict(features)
    status = "Irrigation ON" if prediction[0] > 0.5 else "Irrigation OFF"
    return status

	## preprocess.py

	
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def preprocess_input(soil_moisture: float, temperature: float):
    """
    Normalize input features before feeding into the model.
    """
    data = np.array([[soil_moisture, temperature]])
    scaled = scaler.fit_transform(data)  # for MVP, fit on input directly
    return scaled

	## app.py

	
import gradio as gr
from model import load_model, predict_irrigation
from preprocess import preprocess_input

# Load trained LSTM model
model = load_model()

def irrigation_predict(soil_moisture, temperature):
    features = preprocess_input(soil_moisture, temperature)
    result = predict_irrigation(model, features)
    return result

	### Gradio UI
	
demo = gr.Interface(
    fn=irrigation_predict,
    inputs=[
        gr.Slider(0, 100, step=1, label="Soil Moisture (%)"),
        gr.Slider(-10, 50, step=0.5, label="Temperature (°C)")
    ],
    outputs="text",
    title="Bhu-Rakshak: Smart Irrigation MVP",
    description="Enter soil moisture and temperature to predict irrigation status."
)

if __name__ == "__main__":
    demo.launch()

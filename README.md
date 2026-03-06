# 🏠 Room Occupancy Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

> A smart room occupancy detection system using IoT sensor data and Random Forest classification. The app presents **two predictions simultaneously** — whether the room is occupied (binary) and the exact number of occupants (multivariate) — a core component of intelligent smart building systems.

**[🚀 Live Demo](https://room-occupancy-predictions-model-hf5kew565q8k3syzbxjp99.streamlit.app/)**

---

## 📊 Model Performance

### Binary Model — Is the room occupied?

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Correct Predictions | 2,997 / 3,039 |
| Misclassifications | 42 |
| Accuracy | ~98.6% |

![Binary Model Confusion Matrix](Project%20Insight%20images/binary-model-cm.png)

---

### Multivariate Model — How many occupants?

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Classes | 0, 2, or 3 occupants |
| Correct Predictions | 2,798 / 3,039 |
| Accuracy | ~92.1% |

![Multivariate Model Confusion Matrix](Project%20Insight%20images/multivariate-model-cm.png)

---

## 🔍 Feature Importance

Light sensors are the strongest predictors of room occupancy — far more significant than temperature or sound.

![Top 10 Most Important Features](Project%20Insight%20images/Top-15-features.png)

**Key Insight:** Light intensity (S1, S2, S3) explains ~68% of the model's predictive power. CO2 levels add another ~18%. Temperature and sound play a minor role.

---

## 📡 Dataset & EDA

**Source:** [UCI Machine Learning Repository — Occupancy Estimation Dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)

Data collected via **IoT sensors** placed in a real room across multiple days.

### Feature Distributions
![Dataset Histograms](Project%20Insight%20images/dataset-histograms.png)

### Sensor Correlation Matrix
![Correlation Matrix](Project%20Insight%20images/dataset-correlation-matrix.png)

### Sensors Used

| Sensor | Measures |
|---|---|
| S1–S4 Temperature | Room temperature (°C) across 4 locations |
| S1–S4 Light | Light intensity (Lux) across 4 locations |
| S1–S4 Sound | Sound levels across 4 locations |
| S5 CO2 | CO2 concentration (ppm) |
| S5 CO2 Slope | Rate of CO2 change |
| S6–S7 PIR | Passive Infrared motion sensors |

**Engineered Features:**
- `hour_sin` / `hour_cos` — Cyclical time encoding
- `S1_Sound_lag` — Lagged sound feature
- `S5_CO2_window` — Rolling CO2 window average

---

## 🧠 How It Works

The app runs **both models simultaneously** on the same sensor input:

**1. Binary Model** — *Is anyone in the room?*
- Classifies occupancy as `0` (empty) or `1` (occupied)
- Near-perfect accuracy — only 42 misclassifications out of 3,039

**2. Multivariate Model** — *How many people are in the room?*
- Classifies occupancy count as `0`, `2`, or `3` occupants
- Handles real-world class imbalance (most readings are 0 occupants)

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/SID1014/Room-Occupancy-Predictions-Model.git
cd Room-Occupancy-Predictions-Model

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
Room-Occupancy-Predictions-Model/
│
├── app.py                              # Streamlit app (both models)
├── Training.ipynb                      # Model training notebook
├── Binary_Model_Pipeline.joblib        # Binary classification model
├── Multi_Modle_Pipeline.joblib         # Multivariate model
├── Occupancy_Estimation.csv            # Dataset
├── training_columns.json               # Feature column config
├── requirements.txt
│
└── Project Insight images/             # EDA & model visualizations
    ├── binary-model-cm.png
    ├── multivariate-model-cm.png
    ├── Top-15-features.png
    ├── dataset-correlation-matrix.png
    └── dataset-histograms.png
```

---

## 🛠️ Tech Stack

- **Models:** Scikit-learn (Random Forest, Pipeline, ColumnTransformer)
- **Interface:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Serialization:** Joblib

---

## 🔮 Future Improvements

- Add real-time sensor data streaming
- Experiment with XGBoost / LightGBM for multivariate model
- Improve 2–3 occupant classification accuracy
- Add SHAP explainability visualizations
- Mobile-friendly smart building dashboard
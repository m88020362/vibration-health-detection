# Vibration Analysis Dashboard

A machine learning and deep learning dashboard built with **Streamlit**, designed for vibration signal analysis and anomaly detection.  
This project demonstrates end-to-end workflow skills, including **data preprocessing, feature extraction, model training, and deployment**.

---

## Features

- **Quick Detection (Random Forest Classifier)**  
  Uses statistical features (RMS, Kurtosis, etc.) for health classification.

- **Deep Analysis (3D CNN with CoordConv)**  
  Leverages frequency-domain features from 3-axis vibration signals for deep learning classification.

- **Anomaly Detection (Isolation Forest / One-Class SVM)**  
  Unsupervised methods for early anomaly detection.

- **Interactive Streamlit Dashboard**  
  Upload a `.txt` file with raw 3-axis vibration signals for automatic analysis.  
  Provides statistical features, visualizations, and model predictions.


---

## Installation & Run

### 1. Clone the repository
```{bash}
git clone https://github.com/m88020362/vibration-health-detection.git
cd vibration-health-detection
```

### 2. Install dependencies
```{bash}
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```{bash}
streamlit run app.py
```

## Example Input Format
"X"       "Y"       "Z"
-0.07806   2.00022  -0.02218
-0.05406   1.98972  -0.00512
...
The system will automatically compute RMS and Kurtosis, then perform analysis.

## Skills Demonstrated

- Feature engineering for time-series vibration data

- Model comparison (traditional ML vs deep learning vs anomaly detection)

- Streamlit dashboard design & deployment

- Docker / Streamlit Cloud deployment for public demo


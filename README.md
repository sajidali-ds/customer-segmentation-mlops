# Customer Segmentation MLOps Pipeline

An end-to-end MLOps project that performs customer segmentation using **RFM (Recency, Frequency, Monetary) analysis** and multiple unsupervised machine learning algorithms. The project automatically selects the best-performing clustering model, tracks experiments with **MLflow**, versions models using the **MLflow Model Registry**, and serves predictions through both a **Flask REST API** and a **Streamlit dashboard**.

---

## Features

- RFM (Recency, Frequency, Monetary) Analysis
- Customer Lifetime Value (CLV) Feature Engineering
- Automatic comparison of multiple clustering algorithms
- Best model selection using Silhouette Score
- MLflow Experiment Tracking
- MLflow Model Registry & Versioning
- Production Model Promotion
- Flask REST API
- Streamlit Interactive Dashboard
- Docker Support
- GitHub Actions CI/CD Pipeline
- Unit Testing with Pytest

---

## Architecture

```
Online Retail Dataset
          │
          ▼
   Data Cleaning
          │
          ▼
 Feature Engineering
          │
          ▼
 Feature Scaling
          │
          ▼
 Model Training
          │
          ▼
 Model Comparison
          │
          ▼
 Best Model Selection
          │
          ▼
   MLflow Tracking
          │
          ▼
  Model Registry
          │
          ▼
 Flask API / Streamlit
```

---

## Tech Stack

### Machine Learning

- Scikit-learn
- Pandas
- NumPy

### Backend

- Flask

### Frontend

- Streamlit

### MLOps

- MLflow
- Docker
- GitHub Actions

### Testing

- Pytest

---

## Project Structure

```
customer-segmentation/
│
├── .github/
│   └── workflows/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── mlflow_tracking.py
│   ├── train.py
│   └── predict.py
│
├── templates/
├── static/
├── tests/
│
├── app.py
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **Online Retail Dataset**.

Place the dataset inside:

```
data/raw/online_retail.csv
```

Dataset Link:

https://www.kaggle.com/datasets/vijayuv/onlineretail

---

## Installation

```bash
git clone https://github.com/sajidali-ds/customer-segmentation-mlops.git

cd customer-segmentation-mlops

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Training

```bash
python src/data_pipeline.py

python src/train.py
```

---

## Run Flask API

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Open

```
http://localhost:8501
```

---

## MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
```

Open

```
http://127.0.0.1:5050
```

---

## API Endpoints

| Method | Endpoint |
|----------|-------------------|
| GET | / |
| GET | /health |
| GET | /api/customers |
| GET | /api/segments |
| POST | /api/predict |
| GET | /api/model-info |

---

## Machine Learning Workflow

- Data Cleaning
- Missing Value Handling
- RFM Feature Engineering
- CLV Calculation
- Feature Scaling
- Model Training
- Model Comparison
- Best Model Selection
- MLflow Experiment Tracking
- Model Registration
- Prediction API Deployment

---

## Algorithms Compared

- K-Means
- Agglomerative Clustering
- Gaussian Mixture Model (GMM)
- DBSCAN

---

## Results

- Best Model: **Agglomerative Clustering**
- Automatic model selection using **Silhouette Score**
- MLflow-based experiment tracking
- Automatic model versioning with Model Registry

---

## Future Improvements

- Airflow Data Pipeline
- Model Drift Detection
- Model Monitoring
- AWS Deployment
- Kubernetes Deployment

---

## Author

**Sajid Ali**

B.Tech CSE (Data Science)

Jamia Millia Islamia

GitHub: https://github.com/sajidali-ds

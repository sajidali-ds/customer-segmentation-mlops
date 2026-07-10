# Customer Segmentation & Personalized Marketing Engine

An end-to-end Machine Learning project that segments customers using RFM (Recency, Frequency, Monetary) analysis and multiple clustering algorithms. The application provides customer segmentation through both a Flask REST API and a Streamlit dashboard. MLflow is used for experiment tracking and model versioning.

---

## Features

- RFM-based Customer Segmentation
- CLV (Customer Lifetime Value) Feature Engineering
- Automatic comparison of multiple clustering algorithms
- Best model selection using Silhouette Score
- MLflow Experiment Tracking
- MLflow Model Registry & Versioning
- Flask REST API
- Streamlit Interactive Dashboard
- Docker Support
- GitHub Actions CI Pipeline

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- Streamlit
- MLflow
- Docker
- GitHub Actions

---

## Project Structure

```
customer-segmentation/
│
├── src/
├── templates/
├── static/
├── tests/
├── data/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Installation

```bash
git clone <repository-url>

cd customer-segmentation

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset

Download the **Online Retail Dataset** from Kaggle and place it inside:

```
data/raw/online_retail.csv
```

Dataset:
https://www.kaggle.com/datasets/vijayuv/onlineretail

---

## Training

```bash
python src/data_pipeline.py

python src/train.py
```

---

## Run Flask Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Open:

```
http://localhost:8501
```

---

## MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050
```

Open:

```
http://127.0.0.1:5050
```

---

## API Endpoints

| Method | Endpoint |
|---------|----------|
| GET | / |
| GET | /health |
| GET | /api/segments |
| GET | /api/customers |
| POST | /api/predict |
| GET | /api/model-info |

---

## Machine Learning Workflow

- Data Cleaning
- RFM Feature Engineering
- CLV Calculation
- Feature Scaling
- Model Training
- Model Comparison
- Best Model Selection
- MLflow Tracking
- Model Deployment

Algorithms Used:

- K-Means
- Agglomerative Clustering
- Gaussian Mixture Model
- DBSCAN

---

## Future Improvements

- Airflow Pipeline
- Model Monitoring
- Drift Detection
- PostgreSQL Integration
- Kubernetes Deployment

---

## Author

Sajid Ali
B.Tech CSE (Data Science)
Jamia Millia Islamia   

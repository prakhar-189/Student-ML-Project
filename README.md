# 🎓 Student Marks Predictor

[![CI](https://github.com/prakhar-189/Student-ML-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/prakhar-189/Student-ML-Project/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🔗 **[Live demo](http://student-marks-predictorweb.us-east-1.elasticbeanstalk.com/predictdata)** — deployed on AWS Elastic Beanstalk · ▶️ **[Watch the demo video](docs/Student%20Predictor%20Live%20Demo.mp4)**

![Student Marks Predictor — live web app](docs/Live%20Website%20Home%20Page.png)

An end-to-end regression ML application that predicts a student's **math score** from demographic and academic features. It pairs a clean, modular training pipeline with **MLflow experiment tracking**, an automated **test + CI** workflow, a **Dockerized Flask service** (HTML form + JSON API), and an AWS Elastic Beanstalk deployment.

---

## 📌 Project Overview

The project demonstrates a realistic ML workflow: ingest structured student data, build a reusable preprocessing pipeline, tune and compare several regression models with cross-validation, track every experiment in MLflow, serialize the winning model, and serve it behind a Flask web app and JSON API. Training is fully separated from inference so the deployed service stays lightweight.

---

## ✨ Features

- **Regression-based score prediction** from gender, race/ethnicity, parental education, lunch type, test-prep course, and reading/writing scores.
- **Reusable preprocessing pipeline** — median imputation + standard scaling for numeric features, most-frequent imputation + one-hot encoding for categoricals (`ColumnTransformer`), serialized as `preprocessor.pkl`.
- **Model selection done right** — five regressors tuned with `GridSearchCV`, the winner chosen by **cross-validated R²** (never by peeking at the test set), with train/test R² tracked to surface over-fitting.
- **MLflow experiment tracking** — every model's params and cv/train/test R² are logged and comparable in the MLflow UI.
- **Web app + JSON API** — a Flask form (`/predictdata`) with input validation and a JSON endpoint (`POST /predict`) for programmatic use, served by Gunicorn.
- **Tested & linted in CI** — `pytest` + `ruff` run on every push/PR via GitHub Actions.
- **Containerized** — a `Dockerfile` runs the whole service anywhere.
- **Modular pipeline** — Data Ingestion → Data Transformation → Model Training → Prediction, with custom logging and exceptions.

---

## 📊 Model Performance

Five regressors were tuned with `GridSearchCV` (cv=3) on an 80/20 split of 1,000 records. The winner is selected by **cross-validated R²**; test R² is reported once for the selected model.

| Model | CV R² | Train R² | Test R² |
|-------|:-----:|:--------:|:-------:|
| **Linear Regression** ✅ | **0.865** | 0.874 | **0.880** |
| Gradient Boosting | 0.852 | 0.894 | 0.873 |
| Random Forest | 0.832 | 0.976 | 0.852 |
| AdaBoost | 0.826 | 0.847 | 0.848 |
| Decision Tree | 0.713 | 1.000 | 0.729 |

**Selected model: Linear Regression** — it has the best cross-validated score *and* generalizes best (test R² 0.880). Note how Decision Tree (train 1.000 / test 0.729) and Random Forest (train 0.976) memorize the training data — exactly the over-fitting that CV-based selection guards against.

_Reproduce these numbers with `python -m src.pipeline.train_pipeline` and inspect the runs with `mlflow ui`._

---

## 📁 Project Structure

```
Student-ML-Project/
├── .ebextensions/
│   └── python.config
├── .github/workflows/
│   └── ci.yml                    # Lint (ruff) + tests (pytest) on push/PR
├── Notebook/
│   ├── Dataset/Student.csv
│   ├── EDA Student Performance.ipynb
│   └── Model Training.ipynb
├── artifacts/                    # Generated: data splits, model.pkl, preprocessor.pkl
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py      # GridSearchCV + CV selection + MLflow logging
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/                    # home.html, index.html
├── tests/                        # pytest: CustomData + Flask app/API
├── app.py                        # Flask app: HTML form + JSON API + validation
├── application.py                # WSGI entrypoint (Elastic Beanstalk)
├── Dockerfile / .dockerignore
├── conftest.py / ruff.toml
├── Procfile / buildspec.yml
├── requirements.txt              # Runtime dependencies
└── requirements-dev.txt          # Training/CI deps (mlflow, pytest, ruff)
```

---

## 🛠️ Tech Stack

**Python 3.11** · **Flask** + **Gunicorn** · **scikit-learn** · **Pandas/NumPy** · **MLflow** · **pytest** + **ruff** · **Docker** · **GitHub Actions** · **AWS Elastic Beanstalk / CodePipeline / CodeBuild**

---

## ▶️ Getting Started

```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate      # (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt -r requirements-dev.txt

# 2. (Optional) Retrain — writes artifacts/ and logs runs to MLflow
python -m src.pipeline.train_pipeline
mlflow ui                                          # view experiments at http://localhost:5000

# 3. Run the web app
python app.py                                      # http://localhost:5000
```

### JSON API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"male","race_ethnicity":"group C","parental_level_of_education":"some college","lunch":"standard","test_preparation_course":"completed","reading_score":74,"writing_score":71}'
# -> {"math_score": 74.69}
```

Invalid input returns a `400` with a clear error message instead of leaking a stack trace.

### Run the checks

```bash
ruff check .
pytest -q
```

### Docker

```bash
docker build -t student-marks-predictor .
docker run -p 5000:5000 student-marks-predictor
```

---

## ☁️ Deployment & CI/CD

The app runs live on **AWS Elastic Beanstalk** (single-instance `t3.micro`, Python 3.11):

**→ [student-marks-predictorweb.us-east-1.elasticbeanstalk.com](http://student-marks-predictorweb.us-east-1.elasticbeanstalk.com/predictdata)**

Gunicorn serves the WSGI app (`Procfile` → `gunicorn application:application`), with the platform configured via `.ebextensions/`. The repo also includes a CodeBuild spec (`buildspec.yml`) for a **GitHub → CodeBuild → Elastic Beanstalk** pipeline, so continuous deployment can be re-enabled at any time. On the code side, a GitHub Actions workflow (`ci.yml`) lints and tests every push.

**Deploy an update manually:** build a clean source bundle from the committed code and upload it in the EB console:

```bash
git archive -o ../student-marks.zip HEAD   # zip only tracked files, at the archive root
# EB console → environment → Upload and deploy → choose the zip → set a version label → Deploy
```

> ℹ️ The demo may occasionally be offline if the environment is paused to avoid cloud costs. The app runs identically via `python app.py` or Docker.

---

## 👤 Author

**Prakhar Srivastava** — Aspiring Data Scientist & ML Engineer | Machine Learning, Deep Learning & Generative AI

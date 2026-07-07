# Production image for the Student Marks Predictor Flask app.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

# Install runtime dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app (see .dockerignore for exclusions). Includes artifacts/ so the
# container serves predictions without retraining.
COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Serve via Gunicorn (application object lives in app.py).
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

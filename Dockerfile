FROM python:3.11-slim

WORKDIR /app

# System deps needed by xgboost / scipy wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models directory is written to at runtime (Stage 3 + 4 training artifacts)
RUN mkdir -p /app/models

ENV FLASK_DEBUG=0
ENV PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "app:app"]

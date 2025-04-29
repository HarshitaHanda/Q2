# ---------- base image ----------
FROM python:3.11-slim

# ---------- system libraries ----------
# Tesseract OCR engine  +  Poppler (for pdf2image)  +  basic build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------- working dir ----------
WORKDIR /app

# ---------- python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- application code ----------
COPY main.py .

# ---------- network ----------
EXPOSE 8000

# ---------- start command ----------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

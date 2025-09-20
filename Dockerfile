FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PORT=8080  # Default; Railway overrides with $PORT

# Install system dependencies: libmagic1 for python-magic, poppler-utils for pdf2image, tesseract-ocr for pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--worker-class", "sync"]
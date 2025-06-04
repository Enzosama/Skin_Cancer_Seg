FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/huggingface /tmp/matplotlib && \
    chmod -R 777 /tmp/huggingface /tmp/matplotlib && \
    mkdir -p static/uploads && chmod -R 777 static/uploads
RUN mkdir -p /app/rag_cache && chmod -R 777 /app/rag_cache

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_HOME=/tmp/huggingface \
    HUGGINGFACE_HUB_CACHE=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["python", "app.py"]
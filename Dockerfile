FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS artifacts
WORKDIR /assets
COPY scripts/fetch_runtime_assets.py ./fetch_runtime_assets.py
COPY deploy/runtime-assets.json ./deploy/runtime-assets.json
RUN python fetch_runtime_assets.py --output /assets/data

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY requirements-react.txt ./
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-react.txt
COPY --chown=appuser:appuser . .
COPY --from=artifacts --chown=appuser:appuser /assets/data/ ./
COPY --from=frontend --chown=appuser:appuser /build/dist ./frontend/dist
ENV HF_HOME=/home/appuser/.cache/huggingface OMP_NUM_THREADS=4
USER appuser
EXPOSE 7860
CMD ["uvicorn", "web_api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers"]

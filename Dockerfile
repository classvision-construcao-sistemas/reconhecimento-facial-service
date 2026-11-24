# Imagem de produção (Railway): Flask + SCRFD/TopoFR via onnxruntime.
FROM python:3.12-slim-trixie

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Modelos baixados e conferidos (SHA-256) no build: o container sobe pronto,
# sem baixar ~190 MB a cada reinício.
ENV MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
RUN python -c "from app.config import settings as s; from app.util.model_downloader import garantir_modelo as g; \
g(s.MODELS_DIR, s.SCRFD_ONNX_FICHEIRO, s.SCRFD_ZIP_URL, s.SCRFD_ONNX_SHA256, membro_zip=s.SCRFD_ONNX_FICHEIRO); \
g(s.MODELS_DIR, s.TOPOFR_ONNX_FICHEIRO, s.TOPOFR_ONNX_URL, s.TOPOFR_ONNX_SHA256)"

# O Railway informa a porta em PORT (padrão local: 5000).
EXPOSE 5000
CMD ["python", "main.py"]

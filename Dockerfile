FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para algunas librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto
COPY . .

# Instalar el paquete local para que funcionen imports como serving y core
RUN pip install -e .

# Variables por defecto para cargar el modelo publicado
ENV MODEL_PATH=deployment/model/model.joblib
ENV METADATA_PATH=deployment/model/metadata.json

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
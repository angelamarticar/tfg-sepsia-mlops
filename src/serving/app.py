from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from serving.inference import load_predictor_from_env, SepsisPredictor

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Estado global de la aplicación
# ---------------------------------------------------------------------------

predictor: SepsisPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar y lo libera al apagar."""
    global predictor
    logger.info("Arrancando API. Cargando modelo...")
    predictor = load_predictor_from_env()
    logger.info("Modelo cargado. API lista.")
    yield
    logger.info("Apagando API.")
    predictor = None


app = FastAPI(
    title="SEPSIA — API de detección de sepsis",
    description="Endpoint de inferencia para el modelo de detección temprana de sepsis.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Registro clínico de un paciente en un instante de tiempo."""
    records: list[dict]


class PredictionResult(BaseModel):
    """Resultado de la predicción para un registro."""
    patient_id: str | None = None
    iculos: float | None = None
    sepsis_probability: float
    threshold: float
    alert: bool
    risk_level: str
    model_name: str


class PredictResponse(BaseModel):
    """Respuesta completa del endpoint de predicción."""
    predictions: list[PredictionResult]
    total_records: int
    total_alerts: int


class HealthResponse(BaseModel):
    """Estado del servicio."""
    status: str
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Comprueba que el servicio está activo y el modelo cargado."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor else None,
        threshold=predictor.threshold if predictor else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Genera predicciones de riesgo de sepsis para uno o varios registros clínicos.

    Acepta una lista de registros en formato JSON, cada uno con las variables
    clínicas requeridas por el modelo, y devuelve la probabilidad estimada,
    la alerta binaria y el nivel de riesgo para cada uno.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")

    if not request.records:
        raise HTTPException(status_code=422, detail="La lista de registros está vacía.")

    try:
        results = predictor.predict_many(request.records)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error inesperado durante la inferencia: %s", e)
        raise HTTPException(status_code=500, detail="Error interno durante la inferencia.")

    predictions = [PredictionResult(**r) for r in results]
    total_alerts = sum(1 for p in predictions if p.alert)

    logger.info(
        "Solicitud procesada: %d registros | %d alertas disparadas.",
        len(predictions),
        total_alerts,
    )

    return PredictResponse(
        predictions=predictions,
        total_records=len(predictions),
        total_alerts=total_alerts,
    )
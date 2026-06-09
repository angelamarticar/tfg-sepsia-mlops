# SEPSIA: Sistema de Evaluación Predictiva de Sepsis con Inteligencia Artificial

Trabajo de Fin de Grado centrado en el desarrollo de un prototipo predictivo para la
detección temprana de sepsis a partir de datos clínicos temporales del
PhysioNet/Computing in Cardiology Challenge 2019.

## Objetivo

Desarrollar y evaluar modelos de aprendizaje automático supervisado para predecir la
presencia de sepsis, incorporando un flujo reproducible de preprocesamiento,
entrenamiento, evaluación y trazabilidad inspirado en principios MLOps.

## Requisitos

Python 3.10+

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Los datos se gestionan con DVC. Para descargar los datos versionados:

```bash
dvc pull
```

## Estructura del repositorio

```text
.
├── .dockerignore      # Archivos excluidos durante la construcción de la imagen Docker
├── .gitignore         # Archivos y directorios excluidos del control de versiones con Git
├── Dockerfile         # Definición de la imagen Docker para el despliegue de la API
├── dvc.yaml           # Definición del pipeline reproducible con DVC
├── dvc.lock           # Versiones concretas de datos y artefactos del pipeline
├── params.yaml        # Configuración centralizada de rutas, parámetros y modelos
├── pyproject.toml     # Configuración del proyecto Python
├── requirements.txt   # Dependencias necesarias para ejecutar el proyecto
├── data/
├── deployment/
├── notebooks/
├── reports/
├── scripts/
└── src/

data/
├── raw/          # Datos originales descargados de PhysioNet
├── interim/      # Dataset integrado a partir de los archivos originales
└── processed/    # Datasets preprocesados para modelado y metadatos de preprocesamiento

deployment/
└── model/        # Artefactos del modelo final para inferencia

notebooks/        # Notebooks experimentales
├── 01_exploratory_analysis.ipynb
├── 02_preprocessing_decisiones.ipynb
└── 03_model_training.ipynb

reports/
├── figures/      # Figuras generadas durante la evaluación
├── metrics/
│   ├── evaluate/ # Registro de resultados de evaluación           
│   └── train/    # Registro de resultados de entrenamiento 
├── predictions/  # Registro de predicciones OOF y en test
└── tables/       # Tablas exportadas para la memoria

scripts/
├── download_physionet_data.py   # Descargar conjuntos A y B de Physionet
├── evaluate.py                  # Evalúa los modelos y genera métricas 
├── test_api_request.py          # Prueba de la API de inferencia
└── train.py                     # Ajusta, entrena y compara los modelos

src/
├── core/
│   ├── metrics.py        # Cálculo de métricas de rendimiento
│   ├── plots.py          # Generación de figuras y visualizaciones
│   └── utils.py          # Funciones auxiliares de lectura y guardado
├── data/
│   ├── build_dataset.py        # Construye el dataset integrado a partir de los conjuntos A y B
│   ├── make_scaled_dataset.py  # Aplica escalado robusto a los conjuntos de entrenamiento y test
│   └── preprocess.py           # Aplica preprocesamiento y genera los conjuntos de entrenamiento y test
└── serving/
    ├── app.py            # Define la API y sus endpoints
    └── inference.py      # Carga el modelo final y genera predicciones
    

``` 

## Flujo de datos

El script de descarga no forma parte del pipeline DVC; se utiliza únicamente para obtener los datos originales cuando no están disponibles localmente. El pipeline reproducible comienza a partir de los conjuntos almacenados en `data/raw/`.

Descarga inicial de datos, si no existen en local:
```text
scripts/download_physionet_data.py
        ↓
data/raw/training_setA/ + data/raw/training_setB/
        ↓
src/data/build_dataset.py
        ↓
data/interim/physionet_sepsis_combined.parquet
```

Flujo de datos a través del pipeline DVC:
```text
data/raw/training_setA/ + data/raw/training_setB/
        ↓
src/data/build_dataset.py
        ↓
data/interim/physionet_sepsis_combined.parquet
        ↓
src/data/preprocess.py
        ↓
data/processed/train_preprocessed.parquet
data/processed/test_preprocessed.parquet
        ↓
src/data/make_scaled_dataset.py
        ↓
data/processed/train_preprocessed_scaled.parquet
data/processed/test_preprocessed_scaled.parquet
        ↓
scripts/train.py
        ↓
models/
reports/metrics/train/
reports/predictions/
        ↓
scripts/evaluate.py
        ↓
deployment/model/
reports/metrics/evaluate/
reports/figures/
reports/tables/
```

## Preprocesamiento

El preprocesamiento incluye:

- Particionado train/test a nivel de paciente, estratificado por presencia de sepsis
- Eliminación de variables con ausencia extrema o redundancia clínica
- Creación de indicadores de ausencia para variables con alto porcentaje de valores nulos
- Imputación temporal mediante *forward fill* por paciente
- Imputación final con medianas calculadas exclusivamente sobre entrenamiento
- Recorte (*clipping*) de valores fisiológicamente poco plausibles
- Generación de características temporales mediante ventanas retrospectivas

El escalado es un paso opcional e independiente, ejecutado únicamente para los modelos que lo requieren.

## Ejecución

La ejecución recomendada del proyecto se realiza mediante DVC, utilizando el pipeline definido en `dvc.yaml` y la configuración centralizada en `params.yaml`.

Para reproducir el pipeline completo:
```bash
dvc repro
```
Este comando ejecuta automáticamente las etapas necesarias en función de los cambios detectados en datos, código o parámetros. Las etapas que se ejecutan dependen de las dependencias declaradas en `dvc.yaml`.

También es posible reproducir etapas concretas del pipeline:
```bash
dvc repro build_dataset
dvc repro preprocess
dvc repro make_scaled_dataset
dvc repro train
dvc repro evaluate
```

El flujo completo definido en DVC sigue la siguiente secuencia:
```text

+----------------------------+         +----------------------------+
| data/raw/training_setA.dvc |         | data/raw/training_setB.dvc |
+----------------------------+         +----------------------------+
                       ****                ****
                           ***          ***
                              **      **
                          +---------------+
                          | build_dataset |
                          +---------------+
                                  *
                                  *
                                  *
                            +------------+
                            | preprocess |
                            +------------+
                          ***            ***
                        **                  **
                      **                      **
        +---------------------+                 **
        | make_scaled_dataset |               **
        +---------------------+             **
                          ***            ***
                             **        **
                               **    **
                              +-------+
                              | train |
                              +-------+
                                  *
                                  *
                                  *
                            +----------+
                            | evaluate |
                            +----------+

```

En caso de querer ejecutar los scripts del pipeline de forma aislada, puede hacerse indicando el archivo de parámetros del proyecto:
```bash
python src/data/build_dataset.py --params params.yaml
python src/data/preprocess.py --params params.yaml
python src/data/make_scaled_dataset.py --params params.yaml
python scripts/train.py --params params.yaml
python scripts/evaluate.py --params params.yaml
```

## Versionado de datos, código y experimentos

El código fuente del proyecto se versiona mediante Git y se mantiene en este repositorio de GitHub.

Los datos y artefactos generados se versionan mediante DVC. Los datos originales se referencian mediante archivos `.dvc`, mientras que los datasets intermedios, datasets procesados, modelos, predicciones y reportes se gestionan como salidas del pipeline definido en `dvc.yaml` y registrado en `dvc.lock`. De este modo, los archivos grandes no se almacenan directamente en Git.

La configuración de los experimentos se centraliza en `params.yaml`, lo que permite modificar rutas, parámetros de preprocesamiento, entrenamiento y evaluación sin cambiar directamente el código.

Los experimentos de entrenamiento se registran adicionalmente con MLflow, almacenando métricas, parámetros y artefactos asociados a cada ejecución. Además, el modelo final seleccionado se registra en el MLflow Model Registry.

## Despliegue de la API mediante Docker

El proyecto incluye una API desarrollada con FastAPI para simular el despliegue del modelo final.  La API carga los artefactos publicados en `deployment/model/`, generados durante la etapa `evaluate` del pipeline DVC.

Para construir la imagen Docker:
```bash
docker build -t sepsis-api .
```
Para ejecutar el contenedor:
```bash
docker run -p 8000:8000 sepsis-api
```
Una vez levantada la API, se puede consultar la documentación interactiva en:
```text
http://localhost:8000/docs
```
También puede probarse la inferencia mediante el script incluido en el repositorio:
```bash
python scripts/test_api_request.py
```

El flujo del despliegue simulado es:
```text
scripts/evaluate.py
        ↓
deployment/model/model.joblib
deployment/model/metadata.json
        ↓
src/serving/inference.py
        ↓
src/serving/app.py
        ↓
API FastAPI en Docker
```

La API no constituye un despliegue clínico real, sino una simulación técnica orientada a validar la integración del modelo en un entorno de inferencia reproducible.

## Estado del proyecto

- [x] Descarga e integración de datos
- [x] Análisis exploratorio
- [x] Preprocesamiento reproducible
- [x] Entrenamiento de modelos
- [x] Evaluación del modelo final
- [x] Pipeline MLOps con DVC y MLflow
- [x] Despliegue simulado mediante FastAPI y Docker

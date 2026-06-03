# SEPSIA: Sistema de Evaluación Predictiva de Sepsis con Inteligencia Artificial

Trabajo de Fin de Grado centrado en el desarrollo de un prototipo predictivo para la
detección temprana de sepsis a partir de datos clínicos temporales del
PhysioNet/Computing in Cardiology Challenge 2019.

## Objetivo

Desarrollar y evaluar modelos de aprendizaje automático supervisado para predecir la
presencia de sepsis, incorporando un flujo reproducible de preprocesamiento,
entrenamiento, evaluación y trazabilidad inspirado en principios MLOps.

## Requisitos

Python 3.11+

Instalar dependencias:
- Nota: Falta definir requirements.txt
```bash
pip install -r requirements.txt
```

Los datos se gestionan con DVC. Para descargar los datos versionados:

```bash
dvc pull
```

## Estructura del repositorio

```text
data/
├── raw/          # Datos originales descargados de PhysioNet
├── interim/      # Dataset integrado a partir de los archivos originales
└── processed/    # Datasets preprocesados para modelado

notebooks/
├── 01_analisis_exploratorio.ipynb
└── 02_preprocessing_decisiones.ipynb

src/
├── data/
│   ├── build_dataset.py
│   ├── preprocess.py
│   └── make_scaled_dataset.py
└── models/

reports/
├── figures/
└── metrics/
```

## Flujo de datos

```text
training_setA/ + training_setB/
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
src/data/make_scaled_dataset.py  (solo para modelos que requieren escalado)
        ↓
data/processed/train_preprocessed_scaled.parquet
data/processed/test_preprocessed_scaled.parquet
```

## Preprocesamiento

El preprocesamiento incluye:

- Particionado train/test a nivel de paciente estratificado por presencia de sepsis
- Eliminación de variables con ausencia extrema o redundancia clínica
- Creación de indicadores de ausencia para variables con alto porcentaje de nulos
- Imputación temporal mediante forward fill por paciente
- Imputación final con medianas calculadas exclusivamente sobre entrenamiento
- Clipping de valores fisiológicamente poco plausibles
- Generación de características temporales mediante ventanas retrospectivas

El escalado es un paso opcional ejecutado únicamente para los modelos que lo requieren.

## Ejecución

Construir el dataset integrado:

```bash
python src/data/build_dataset.py \
  --input-a data/raw/training_setA \
  --input-b data/raw/training_setB \
  --output data/interim/physionet_sepsis_combined.parquet
```

Aplicar preprocesamiento base:

```bash
python src/data/preprocess.py \
  --input data/interim/physionet_sepsis_combined.parquet \
  --output-dir data/processed
```

Generar datasets escalados (opcional):

```bash
python src/data/make_scaled_dataset.py \
  --train data/processed/train_preprocessed.parquet \
  --test data/processed/test_preprocessed.parquet \
  --output-dir data/processed
```

## Versionado de datos

Los datos se versionan mediante DVC. Los archivos grandes no se almacenan
directamente en Git, sino a través de archivos `.dvc`.

## Estado del proyecto

- [x] Descarga e integración de datos
- [x] Análisis exploratorio
- [x] Preprocesamiento reproducible
- [ ] Entrenamiento de modelos
- [ ] Evaluación del modelo final
- [ ] Pipeline MLOps
- [ ] Despliegue simulado

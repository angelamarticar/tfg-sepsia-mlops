import json
import requests
import pandas as pd
from pathlib import Path


API_URL = "http://127.0.0.1:8000/predict"

METADATA_PATH = Path("deployment/model/metadata.json")
TEST_PATH = Path("data/processed/test_preprocessed.parquet")


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["params"]["feature_cols"]

    df_test = pd.read_parquet(TEST_PATH)

    cols_to_send = feature_cols.copy()

    for col in ["PatientID", "ICULOS"]:
        if col in df_test.columns and col not in cols_to_send:
            cols_to_send.append(col)

    # Para la captura, mejor solo 1 registro
    sample = df_test.sample(n=1, random_state=42).copy()

    # Convertir NaN a None para que sea JSON válido
    sample = sample.where(pd.notnull(sample), None)

    records = sample[cols_to_send].to_dict(orient="records")

    payload = {
        "records": records
    }

    print("JSON para pegar en Swagger:")
    print(json.dumps(payload, indent=4, ensure_ascii=False))

    response = requests.post(API_URL, json=payload, timeout=30)

    print("\nStatus code:", response.status_code)
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
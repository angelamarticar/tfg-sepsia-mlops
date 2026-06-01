import os
import re
from urllib.parse import urljoin

import requests


BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"


def download_training_set(set_name: str, output_root: str = "data/raw") -> None:
    """
    Descarga los archivos .psv de un conjunto de entrenamiento del
    PhysioNet/Computing in Cardiology Challenge 2019.

    Parameters
    ----------
    set_name : str
        Nombre del conjunto a descargar. Por ejemplo: "training_setA" o "training_setB".
    output_root : str
        Directorio raíz donde se almacenarán los datos descargados.
    """
    base_url = urljoin(BASE_URL, f"{set_name}/")
    dest_dir = os.path.join(output_root, set_name)

    os.makedirs(dest_dir, exist_ok=True)

    response = requests.get(base_url, timeout=30)
    response.raise_for_status()

    files = sorted(set(re.findall(r'href="([^"]+\.psv)"', response.text)))

    print(f"[{set_name}] Encontrados {len(files)} archivos .psv")

    for i, filename in enumerate(files, 1):
        file_url = urljoin(base_url, filename)
        output_path = os.path.join(dest_dir, filename)

        # Evita descargar de nuevo archivos que ya existen
        if os.path.exists(output_path):
            print(f"[{i}/{len(files)}] Ya existe {filename}, se omite")
            continue

        file_response = requests.get(file_url, timeout=60)
        file_response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(file_response.content)

        print(f"[{i}/{len(files)}] Descargado {filename}")


def main() -> None:
    download_training_set("training_setA")
    download_training_set("training_setB")


if __name__ == "__main__":
    main()
"""
src/data/ingestion.py
─────────────────────
Extrae un archivo .zip de imágenes de aves y lo organiza en data/raw/
manteniendo la estructura de carpetas por clase:

    dataset.zip
    ├── ESPECIE_A/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── ESPECIE_B/
        └── img3.jpg

    →  data/raw/
       ├── ESPECIE_A/
       │   ├── img1.jpg
       │   └── img2.jpg
       └── ESPECIE_B/
           └── img3.jpg
"""

import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Optional

import yaml

import hashlib
from PIL import Image
import io


# ── Extensiones reconocidas como imagen ──────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_config(config_path: str = "configs/dataset.yaml") -> dict:
    """Carga la configuración desde el archivo YAML del proyecto."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def convert_to_jpg(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        return output.getvalue()

def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:10]
    
def ingest_zip(
    zip_path: str,
    config_path: str = "configs/dataset.yaml",
    raw_data_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Lee un .zip, extrae las carpetas de clases en data/raw/ y reporta
    cuántas imágenes encontró por clase.

    Args:
        zip_path      : Ruta al archivo .zip con el dataset.
        config_path   : Ruta al YAML de configuración del proyecto.
        raw_data_path : Sobrescribe `raw_data_path` del YAML si se indica.
        verbose       : Si True, imprime el reporte en consola.

    Returns:
        dict con las claves:
            "raw_dir"       : Path — carpeta de destino
            "classes"       : list[str] — nombres de clases encontradas
            "counts"        : dict[str, int] — imágenes por clase
            "total_images"  : int — total de imágenes extraídas
            "skipped_files" : list[str] — archivos ignorados (no imagen / sin clase)
    """
    # ── Configuración ────────────────────────────────────────────
    config       = load_config(config_path)
    raw_dir      = Path(raw_data_path or config["raw_data_path"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"El archivo no es un .zip válido: {zip_path}")

    # ── Extracción selectiva ──────────────────────────────────────
    counts: dict[str, int] = defaultdict(int)
    skipped: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        for member in members:
            member_path = Path(member.filename)

            # Ignorar directorios y archivos en la raíz sin carpeta de clase
            if member.is_dir():
                continue
            parts = member_path.parts
            if len(parts) < 2:
                skipped.append(member.filename)
                continue

            # La primera parte es el nombre de la clase (carpeta raíz)
            # Puede haber un nivel extra si el zip tiene carpeta contenedora
            # Detectamos si parts[0] es una carpeta "contenedora" sin imágenes
            class_name = parts[0]
            file_name  = member_path.name

            # Ignorar archivos que no sean imágenes
            if member_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append(member.filename)
                continue

            # Destino: raw_dir / clase / imagen
            dest = raw_dir / class_name / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Extraer archivo
            with zf.open(member) as src:
                file_bytes = src.read()
            
                try:
                    # Convertir a JPG
                    jpg_bytes = convert_to_jpg(file_bytes)
            
                    # Generar hash
                    file_hash = hash_bytes(jpg_bytes)
            
                    # Nombre limpio
                    safe_class = class_name.lower().replace(" ", "_")
                    new_name = f"{safe_class}_{file_hash}.jpg"
            
                    dest = raw_dir / class_name / new_name
                    dest.parent.mkdir(parents=True, exist_ok=True)
            
                    # Evitar duplicados
                    if not dest.exists():
                        with open(dest, "wb") as dst:
                            dst.write(jpg_bytes)
                        counts[class_name] += 1
            
                except Exception:
                    skipped.append(member.filename)

            counts[class_name] += 1

    # ── Si todas las clases tienen 0 imágenes, puede haber un nivel extra ──
    # (zip con carpeta raíz contenedora: dataset/ESPECIE_A/img.jpg)
    if not counts:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                parts = member_path.parts
                if len(parts) < 3:
                    skipped.append(member.filename)
                    continue
                if member_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped.append(member.filename)
                    continue

                # parts[0] = carpeta contenedora, parts[1] = clase
                class_name = parts[1]
                file_name  = member_path.name

                dest = raw_dir / class_name / file_name
                dest.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src:
                    file_bytes = src.read()
                
                    try:
                        # Convertir a JPG
                        jpg_bytes = convert_to_jpg(file_bytes)
                
                        # Generar hash
                        file_hash = hash_bytes(jpg_bytes)
                
                        # Nombre limpio
                        safe_class = class_name.lower().replace(" ", "_")
                        new_name = f"{safe_class}_{file_hash}.jpg"
                
                        dest = raw_dir / class_name / new_name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                
                        # Evitar duplicados
                        if not dest.exists():
                            with open(dest, "wb") as dst:
                                dst.write(jpg_bytes)
                            counts[class_name] += 1
                
                    except Exception:
                        skipped.append(member.filename)

                counts[class_name] += 1

    # ── Resultados ───────────────────────────────────────────────
    classes      = sorted(counts.keys())
    total_images = sum(counts.values())

    if verbose:
        _print_report(zip_path, raw_dir, classes, counts, total_images, skipped)

    return {
        "raw_dir":       raw_dir,
        "classes":       classes,
        "counts":        dict(counts),
        "total_images":  total_images,
        "skipped_files": skipped,
    }

def _print_report(
    zip_path: Path,
    raw_dir: Path,
    classes: list,
    counts: dict,
    total_images: int,
    skipped: list,
) -> None:
    """Imprime un reporte legible de la ingestión."""
    sep = "─" * 52
    print(sep)
    print(f"  INGESTION REPORT — W&F BirdLab")
    print(sep)
    print(f"  Origen  : {zip_path}")
    print(f"  Destino : {raw_dir}")
    print(f"  Clases encontradas: {len(classes)}")
    print()
    print(f"  {'Clase':<35} {'Imagenes':>8}")
    print(f"  {'-'*35} {'-'*8}")
    for cls in classes:
        print(f"  {cls:<35} {counts[cls]:>8}")
    print(f"  {'─'*35} {'─'*8}")
    print(f"  {'TOTAL':<35} {total_images:>8}")
    print()
    if skipped:
        print(f"  Archivos ignorados: {len(skipped)}")
        for s in skipped[:5]:
            print(f"    · {s}")
        if len(skipped) > 5:
            print(f"    ... y {len(skipped) - 5} más")
    else:
        print("  Sin archivos ignorados.")
    print(sep)
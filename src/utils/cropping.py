"""
src/data/cropping.py
────────────────────
Pipeline de recorte de aves con clasificación en tres estados.

Estado de cada imagen tras el pipeline:
  ACCEPTED     → imagen válida, guardada en data/processed/
  NEEDS_REVIEW → imagen dudosa, guardada en data/review/ para inspección manual
  REJECTED     → imagen descartada, solo se registra en metadata

Reglas de clasificación
─────────────────────────────────────────────────────────────────
  Sin detección YOLO
      → REJECTED (no_detection)

  confidence < 0.30  OR  bbox_area_ratio < 0.02
      → REJECTED (hard thresholds)

  ANY de las siguientes:
      confidence       < 0.60
      bbox_area_ratio  < 0.08
      bbox toca borde de la imagen
      aspect ratio del crop fuera de [0.5, 2.0]
      bbox requirió clipping
      imagen final < image_size px en algún lado
      → NEEDS_REVIEW

  Todo lo demás
      → ACCEPTED

Regla fundamental: NUNCA se introducen bordes artificiales (cero padding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image


# ══════════════════════════════════════════════════════════════════
#  Constantes de estado y rechazo
# ══════════════════════════════════════════════════════════════════

class ImageStatus:
    ACCEPTED     = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED     = "rejected"


class RejectionReason:
    # Rechazos duros (imagen descartada)
    NO_DETECTION       = "no_detection"
    HARD_LOW_CONF      = "hard_low_confidence"       # conf < 0.30
    HARD_SMALL_BBOX    = "hard_small_bbox"            # area_ratio < 0.02
    BBOX_CLIPPED_EMPTY = "bbox_empty_after_clip"
    LOAD_ERROR         = "load_error"

    # Razones de needs_review (pueden aparecer combinadas)
    SOFT_LOW_CONF      = "soft_low_confidence"        # conf < 0.60
    SOFT_SMALL_BBOX    = "soft_small_bbox"            # area_ratio < 0.08
    TOUCHES_EDGE       = "touches_image_edge"
    BAD_ASPECT_RATIO   = "aspect_ratio_out_of_range"
    REQUIRED_CLIPPING  = "bbox_required_clipping"
    FINAL_TOO_SMALL    = "final_image_too_small"


# ══════════════════════════════════════════════════════════════════
#  Métricas intermedias
# ══════════════════════════════════════════════════════════════════

@dataclass
class DetectionMetrics:
    """Todas las métricas computadas sobre una detección."""
    confidence:      float = 0.0
    bbox_area_ratio: float = 0.0   # área del crop / área imagen original
    touches_edge:    bool  = False  # el bbox (con margen) tocó algún borde
    clipped:         bool  = False  # el bbox fue recortado por algún borde
    aspect_ratio:    float = 1.0   # w/h del crop tras clipping
    final_w:         int   = 0
    final_h:         int   = 0

    # bbox coords
    bbox_raw:     Optional[tuple] = None  # (x1,y1,x2,y2) de YOLO puro
    bbox_margin:  Optional[tuple] = None  # tras aplicar margen
    bbox_clipped: Optional[tuple] = None  # tras clip a bordes

    @property
    def final_size_ok(self) -> bool:
        return self.final_w > 0 and self.final_h > 0


# ══════════════════════════════════════════════════════════════════
#  Resultado del pipeline
# ══════════════════════════════════════════════════════════════════

@dataclass
class CropResult:
    """Resultado completo del pipeline para una imagen."""
    status:          str                        # ImageStatus.*
    image:           Optional[Image.Image] = None
    original_path:   Optional[Path]        = None
    species:         Optional[str]         = None  # nombre de carpeta/clase
    metrics:         Optional[DetectionMetrics] = None
    rejection_reason: Optional[str]        = None  # razón dura si REJECTED
    review_flags:    list[str]             = field(default_factory=list)  # razones si NEEDS_REVIEW

    @property
    def accepted(self) -> bool:
        return self.status == ImageStatus.ACCEPTED

    @property
    def needs_review(self) -> bool:
        return self.status == ImageStatus.NEEDS_REVIEW

    @property
    def rejected(self) -> bool:
        return self.status == ImageStatus.REJECTED

    def to_metadata_dict(self) -> dict:
        """Serializa a dict apto para guardar en SQLite via metadata.py."""
        m = self.metrics
        return {
            "image":          self.original_path.name if self.original_path else None,
            "class":          self.species,
            "status":         self.status,
            "confidence":     round(m.confidence, 4)     if m else None,
            "bbox_area_ratio":round(m.bbox_area_ratio, 4) if m else None,
            "touches_edge":   m.touches_edge              if m else None,
            "clipped":        m.clipped                   if m else None,
            "aspect_ratio":   round(m.aspect_ratio, 4)   if m else None,
            "final_w":        m.final_w                   if m else None,
            "final_h":        m.final_h                   if m else None,
            "rejection_reason": self.rejection_reason,
            "review_flags":   ",".join(self.review_flags) if self.review_flags else None,
        }


# ══════════════════════════════════════════════════════════════════
#  Configuración
# ══════════════════════════════════════════════════════════════════

@dataclass
class CropConfig:
    image_size:          int   = 224
    margin_ratio:        float = 0.20

    # Umbrales duros (REJECTED si no se cumplen)
    hard_min_confidence:      float = 0.30
    hard_min_bbox_area_ratio: float = 0.02

    # Umbrales blandos (NEEDS_REVIEW si alguno falla)
    soft_min_confidence:      float = 0.60
    soft_min_bbox_area_ratio: float = 0.08
    min_aspect_ratio:         float = 0.50   # w/h mínimo del crop
    max_aspect_ratio:         float = 2.00   # w/h máximo del crop

    @classmethod
    def from_yaml(cls, config_path: str = "configs/dataset.yaml") -> "CropConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config no encontrada: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Compatibilidad con versiones previas del YAML
        # min_confidence y min_bbox_area_ratio del YAML original se
        # mapean a los umbrales blandos.
        soft_conf = cfg.get("soft_min_confidence",
                    cfg.get("min_confidence", 0.60))
        soft_area = cfg.get("soft_min_bbox_area_ratio",
                    cfg.get("min_bbox_area_ratio", 0.08))

        return cls(
            image_size               = cfg.get("image_size",              224),
            margin_ratio             = cfg.get("margin_ratio",             0.20),
            hard_min_confidence      = cfg.get("hard_min_confidence",      0.30),
            hard_min_bbox_area_ratio = cfg.get("hard_min_bbox_area_ratio", 0.02),
            soft_min_confidence      = soft_conf,
            soft_min_bbox_area_ratio = soft_area,
            min_aspect_ratio         = cfg.get("min_aspect_ratio",         0.50),
            max_aspect_ratio         = cfg.get("max_aspect_ratio",         2.00),
        )


# ══════════════════════════════════════════════════════════════════
#  Cropper principal
# ══════════════════════════════════════════════════════════════════

class BirdCropper:
    """
    Recorta aves a partir de detecciones YOLO con clasificación en
    tres estados: ACCEPTED / NEEDS_REVIEW / REJECTED.

    Uso:
        cropper = BirdCropper.from_config("configs/dataset.yaml")
        result  = cropper.crop(Path("data/raw/ESPECIE_A/img.jpg"))
        if result.accepted:
            result.image.save("salida.jpg")
        print(result.to_metadata_dict())
    """

    BIRD_CLASS_ID = 14   # clase "bird" en COCO

    def __init__(self, config: CropConfig):
        self.cfg    = config
        self._model = None

    @classmethod
    def from_config(cls, config_path: str = "configs/dataset.yaml") -> "BirdCropper":
        return cls(CropConfig.from_yaml(config_path))

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
        return self._model

    # ── Pipeline principal ────────────────────────────────────────

    def crop(self, image_path: Path, species: str = None) -> CropResult:
        """
        Ejecuta el pipeline completo.

        Args:
            image_path : ruta a la imagen original
            species    : nombre de la clase/especie (tomado de la carpeta si None)

        Returns:
            CropResult con status, image (si no rechazada), métricas y flags.
        """
        image_path = Path(image_path)
        if species is None:
            species = image_path.parent.name

        def _reject(reason: str, metrics: DetectionMetrics = None) -> CropResult:
            return CropResult(
                status=ImageStatus.REJECTED,
                original_path=image_path,
                species=species,
                metrics=metrics,
                rejection_reason=reason,
            )

        # ── 1. Cargar imagen ──────────────────────────────────────
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return _reject(RejectionReason.LOAD_ERROR)

        W, H = img.size

        # ── 2. Detección YOLO ─────────────────────────────────────
        yolo_results = self.model(img, verbose=False)
        best = self._best_bird_detection(yolo_results)

        if best is None:
            return _reject(RejectionReason.NO_DETECTION)

        conf, (bx1, by1, bx2, by2) = best

        # ── 3. Computar todas las métricas ────────────────────────
        metrics = DetectionMetrics(
            confidence  = conf,
            bbox_raw    = (bx1, by1, bx2, by2),
        )

        # ── 4. Umbrales DUROS ─────────────────────────────────────
        # Antes de aplicar margen, calculamos bbox_area_ratio sobre el
        # bbox YOLO original (sin margen) para el filtro duro.
        raw_area_ratio = ((bx2 - bx1) * (by2 - by1)) / (W * H)
        metrics.bbox_area_ratio = raw_area_ratio

        if conf < self.cfg.hard_min_confidence:
            return _reject(RejectionReason.HARD_LOW_CONF, metrics)

        if raw_area_ratio < self.cfg.hard_min_bbox_area_ratio:
            return _reject(RejectionReason.HARD_SMALL_BBOX, metrics)

        # ── 5. Margen dinámico ────────────────────────────────────
        bw = bx2 - bx1
        bh = by2 - by1
        mx = bw * self.cfg.margin_ratio
        my = bh * self.cfg.margin_ratio

        ex1 = bx1 - mx
        ey1 = by1 - my
        ex2 = bx2 + mx
        ey2 = by2 + my

        metrics.bbox_margin = (ex1, ey1, ex2, ey2)

        # ── 6. Clip a límites (sin padding) ───────────────────────
        cx1 = max(0.0, ex1)
        cy1 = max(0.0, ey1)
        cx2 = min(float(W), ex2)
        cy2 = min(float(H), ey2)

        if cx2 <= cx1 or cy2 <= cy1:
            return _reject(RejectionReason.BBOX_CLIPPED_EMPTY, metrics)

        metrics.bbox_clipped = (cx1, cy1, cx2, cy2)

        # ¿hubo clipping en algún borde?
        metrics.clipped     = (cx1 > ex1 or cy1 > ey1 or
                                cx2 < ex2 or cy2 < ey2)
        # ¿el bbox (tras clip) toca algún borde de la imagen?
        metrics.touches_edge = (cx1 == 0 or cy1 == 0 or
                                 cx2 == float(W) or cy2 == float(H))

        # Recalcular area_ratio sobre la región clipeada
        crop_area            = (cx2 - cx1) * (cy2 - cy1)
        metrics.bbox_area_ratio = crop_area / (W * H)
        metrics.aspect_ratio = (cx2 - cx1) / (cy2 - cy1)

        # ── 7. Recortar ───────────────────────────────────────────
        cropped = img.crop((int(cx1), int(cy1), int(cx2), int(cy2)))

        # ── 8. Resize preservando aspect ratio ───────────────────
        resized = self._resize_keep_ratio(cropped, self.cfg.image_size)
        rw, rh  = resized.size
        metrics.final_w = rw
        metrics.final_h = rh

        # ── 9. Center crop ────────────────────────────────────────
        if rw >= self.cfg.image_size and rh >= self.cfg.image_size:
            final = self._center_crop(resized, self.cfg.image_size)
            metrics.final_w = self.cfg.image_size
            metrics.final_h = self.cfg.image_size
        else:
            final = resized   # guardar tal cual; el flag lo marca

        # ── 10. Clasificar: NEEDS_REVIEW o ACCEPTED ───────────────
        review_flags = self._compute_review_flags(metrics)

        if review_flags:
            status = ImageStatus.NEEDS_REVIEW
        else:
            status = ImageStatus.ACCEPTED

        return CropResult(
            status        = status,
            image         = final,
            original_path = image_path,
            species       = species,
            metrics       = metrics,
            review_flags  = review_flags,
        )

    # ── Clasificación de flags ────────────────────────────────────

    def _compute_review_flags(self, m: DetectionMetrics) -> list[str]:
        """
        Devuelve lista de razones por las que una imagen va a NEEDS_REVIEW.
        Lista vacía → ACCEPTED.
        """
        flags = []

        if m.confidence < self.cfg.soft_min_confidence:
            flags.append(RejectionReason.SOFT_LOW_CONF)

        if m.bbox_area_ratio < self.cfg.soft_min_bbox_area_ratio:
            flags.append(RejectionReason.SOFT_SMALL_BBOX)

        if m.touches_edge:
            flags.append(RejectionReason.TOUCHES_EDGE)

        if not (self.cfg.min_aspect_ratio
                <= m.aspect_ratio
                <= self.cfg.max_aspect_ratio):
            flags.append(RejectionReason.BAD_ASPECT_RATIO)

        if m.clipped:
            flags.append(RejectionReason.REQUIRED_CLIPPING)

        if m.final_w < self.cfg.image_size or m.final_h < self.cfg.image_size:
            flags.append(RejectionReason.FINAL_TOO_SMALL)

        return flags

    # ── Helpers ───────────────────────────────────────────────────

    def _best_bird_detection(self, results) -> Optional[tuple]:
        """
        Devuelve (conf, (x1,y1,x2,y2)) de la detección de mayor confianza
        que sea clase bird (COCO 14) o cualquier clase en modelos especializados.
        """
        best_conf = -1.0
        best_box  = None

        for result in results:
            if result.boxes is None:
                continue
            single_class_model = len(result.names) == 1
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())

                is_bird = (
                    single_class_model
                    or cls_id == self.BIRD_CLASS_ID
                    or result.names.get(cls_id, "").lower()
                    in {"bird", "ave", "pajaro"}
                )
                if not is_bird:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].tolist()
                    best_box = (float(xyxy[0]), float(xyxy[1]),
                                float(xyxy[2]), float(xyxy[3]))

        return None if best_box is None else (best_conf, best_box)

    @staticmethod
    def _resize_keep_ratio(img: Image.Image, target: int) -> Image.Image:
        """Redimensiona para que el lado MENOR sea exactamente `target`."""
        w, h = img.size
        if w < h:
            new_w, new_h = target, round(h * target / w)
        else:
            new_w, new_h = round(w * target / h), target
        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _center_crop(img: Image.Image, size: int) -> Image.Image:
        w, h   = img.size
        left   = (w - size) // 2
        top    = (h - size) // 2
        return img.crop((left, top, left + size, top + size))


# ══════════════════════════════════════════════════════════════════
#  Función de alto nivel para procesar una carpeta de clase
# ══════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def process_class_directory(
    class_dir:    Path,
    accepted_dir: Path,
    review_dir:   Path,
    cropper:      BirdCropper,
) -> dict:
    """
    Procesa todas las imágenes de una carpeta de clase.

    Guarda:
      - ACCEPTED     → accepted_dir / clase / imagen
      - NEEDS_REVIEW → review_dir   / clase / imagen
      - REJECTED     → solo metadata, no se guarda en disco

    Returns:
        {
          "accepted":  int,
          "needs_review": int,
          "rejected":  int,
          "total":     int,
          "results":   list[CropResult]   ← para guardar en metadata
        }
    """
    accepted_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    counts  = {"accepted": 0, "needs_review": 0, "rejected": 0}
    results = []

    images = sorted(
        f for f in class_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in images:
        result = cropper.crop(img_path, species=class_dir.name)
        results.append(result)

        if result.accepted:
            dest = accepted_dir / img_path.name
            result.image.save(dest, quality=95)
            counts["accepted"] += 1

        elif result.needs_review:
            dest = review_dir / img_path.name
            result.image.save(dest, quality=95)
            counts["needs_review"] += 1

        else:
            counts["rejected"] += 1

    counts["total"]   = len(images)
    counts["results"] = results
    return counts
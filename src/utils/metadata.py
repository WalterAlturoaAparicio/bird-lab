"""
src/data/metadata.py
────────────────────
Gestión de la base de datos SQLite que registra el resultado del
pipeline de preprocesamiento para cada imagen.

Esquema de la tabla `images`:
    image           TEXT       — nombre del archivo (bird_123.jpg)
    class           TEXT       — nombre de la especie / carpeta
    status          TEXT       — accepted | needs_review | rejected
    confidence      REAL       — confianza de la detección YOLO
    bbox_area_ratio REAL       — área del crop / área imagen original
    touches_edge    INTEGER    — 1 si el bbox tocó algún borde
    clipped         INTEGER    — 1 si el bbox fue recortado
    aspect_ratio    REAL       — w/h del crop tras clipping
    final_w         INTEGER    — ancho de la imagen final (px)
    final_h         INTEGER    — alto de la imagen final (px)
    rejection_reason TEXT      — razón si status=rejected
    review_flags    TEXT       — flags separados por coma si needs_review
    processed_at    TEXT       — timestamp ISO 8601

Uso:
    from src.utils.metadata import MetadataDB
    db = MetadataDB("metadata/metadata.sqlite")
    db.insert(result.to_metadata_dict())
    db.insert_batch([r.to_metadata_dict() for r in results])
    summary = db.summary()
    rejected = db.query_by_status("rejected")
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


# ══════════════════════════════════════════════════════════════════
#  DDL — definición de la tabla
# ══════════════════════════════════════════════════════════════════

_DDL = """
CREATE TABLE IF NOT EXISTS images (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    image            TEXT    NOT NULL,
    class            TEXT,
    status           TEXT    NOT NULL,
    confidence       REAL,
    bbox_area_ratio  REAL,
    touches_edge     INTEGER,
    clipped          INTEGER,
    aspect_ratio     REAL,
    final_w          INTEGER,
    final_h          INTEGER,
    rejection_reason TEXT,
    review_flags     TEXT,
    processed_at     TEXT    NOT NULL,
    UNIQUE(image, class)        -- evitar duplicados por re-ejecución
);

CREATE INDEX IF NOT EXISTS idx_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_class  ON images(class);
"""

_INSERT = """
INSERT OR REPLACE INTO images (
    image, class, status, confidence, bbox_area_ratio,
    touches_edge, clipped, aspect_ratio, final_w, final_h,
    rejection_reason, review_flags, processed_at
) VALUES (
    :image, :class, :status, :confidence, :bbox_area_ratio,
    :touches_edge, :clipped, :aspect_ratio, :final_w, :final_h,
    :rejection_reason, :review_flags, :processed_at
)
"""


# ══════════════════════════════════════════════════════════════════
#  MetadataDB
# ══════════════════════════════════════════════════════════════════

class MetadataDB:
    """
    Interfaz sobre la base de datos SQLite de preprocesamiento.

    Thread-safety: cada llamada abre/cierra su propia conexión con
    check_same_thread=False, apto para uso secuencial en notebooks.
    Para uso concurrente real, usar una conexión por hilo.
    """

    def __init__(self, db_path: str | Path = "metadata/metadata.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Contexto de conexión ──────────────────────────────────────

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Inicialización ────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ── Escritura ─────────────────────────────────────────────────

    def insert(self, record: dict) -> None:
        """
        Inserta o reemplaza un registro.
        `record` es el dict devuelto por CropResult.to_metadata_dict().
        """
        row = self._prepare(record)
        with self._conn() as conn:
            conn.execute(_INSERT, row)

    def insert_batch(self, records: list[dict]) -> int:
        """
        Inserta múltiples registros en una sola transacción.
        Retorna el número de filas insertadas/reemplazadas.
        """
        if not records:
            return 0
        rows = [self._prepare(r) for r in records]
        with self._conn() as conn:
            conn.executemany(_INSERT, rows)
        return len(rows)

    @staticmethod
    def _prepare(record: dict) -> dict:
        """Normaliza un dict antes de insertarlo."""
        row = dict(record)
        row["processed_at"] = datetime.now(timezone.utc).isoformat()
        # Convertir bools a int para SQLite
        for key in ("touches_edge", "clipped"):
            if isinstance(row.get(key), bool):
                row[key] = int(row[key])
        return row

    # ── Lectura ───────────────────────────────────────────────────

    def query_by_status(self, status: str) -> list[dict]:
        """Devuelve todos los registros con el estado dado."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM images WHERE status = ? ORDER BY class, image",
                (status,),
            )
            return [dict(row) for row in cur.fetchall()]

    def query_by_class(self, class_name: str) -> list[dict]:
        """Devuelve todos los registros de una clase."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM images WHERE class = ? ORDER BY status, image",
                (class_name,),
            )
            return [dict(row) for row in cur.fetchall()]

    def query_by_flag(self, flag: str) -> list[dict]:
        """Devuelve registros de needs_review que contengan un flag específico."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM images WHERE review_flags LIKE ? ORDER BY class, image",
                (f"%{flag}%",),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM images ORDER BY class, image")
            return [dict(row) for row in cur.fetchall()]

    # ── Resumen estadístico ───────────────────────────────────────

    def summary(self) -> dict:
        """
        Devuelve estadísticas globales del dataset procesado.

        Returns:
            {
              "total": int,
              "by_status": {"accepted": int, "needs_review": int, "rejected": int},
              "by_class": {class_name: {"accepted":int, "needs_review":int, "rejected":int}},
              "review_flag_counts": {flag: int},
              "rejection_reason_counts": {reason: int},
              "avg_confidence": float,
              "avg_bbox_area_ratio": float,
            }
        """
        with self._conn() as conn:

            # Totales por estado
            cur = conn.execute(
                "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
            )
            by_status = {row["status"]: row["n"] for row in cur.fetchall()}

            total = sum(by_status.values())

            # Por clase y estado
            cur = conn.execute(
                """SELECT class, status, COUNT(*) AS n
                   FROM images GROUP BY class, status"""
            )
            by_class: dict[str, dict] = {}
            for row in cur.fetchall():
                cls = row["class"] or "_unknown"
                by_class.setdefault(cls, {})
                by_class[cls][row["status"]] = row["n"]

            # Conteo de review_flags
            cur = conn.execute(
                "SELECT review_flags FROM images WHERE review_flags IS NOT NULL"
            )
            flag_counts: dict[str, int] = {}
            for row in cur.fetchall():
                for flag in row["review_flags"].split(","):
                    flag = flag.strip()
                    if flag:
                        flag_counts[flag] = flag_counts.get(flag, 0) + 1

            # Conteo de rejection_reasons
            cur = conn.execute(
                """SELECT rejection_reason, COUNT(*) AS n
                   FROM images
                   WHERE rejection_reason IS NOT NULL
                   GROUP BY rejection_reason"""
            )
            rejection_counts = {row["rejection_reason"]: row["n"]
                                for row in cur.fetchall()}

            # Promedios
            cur = conn.execute(
                """SELECT AVG(confidence)      AS avg_conf,
                          AVG(bbox_area_ratio) AS avg_area
                   FROM images WHERE confidence IS NOT NULL"""
            )
            row  = cur.fetchone()
            avgs = dict(row) if row else {}

        return {
            "total":                   total,
            "by_status":               by_status,
            "by_class":                by_class,
            "review_flag_counts":      flag_counts,
            "rejection_reason_counts": rejection_counts,
            "avg_confidence":          round(avgs.get("avg_conf",  0) or 0, 4),
            "avg_bbox_area_ratio":     round(avgs.get("avg_area",  0) or 0, 4),
        }

    # ── Utilidades ────────────────────────────────────────────────

    def clear(self) -> None:
        """Elimina todos los registros (útil para re-procesar desde cero)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM images")

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def __repr__(self) -> str:
        return f"MetadataDB(path={self.db_path}, rows={self.count()})"
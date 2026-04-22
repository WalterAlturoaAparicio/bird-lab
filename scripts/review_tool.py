#!/usr/bin/env python3
"""
W&F BirdLab -- review_tool.py  (v2)
=====================================
Herramienta de revision manual con OpenCV para tres fuentes:

  --source review     Imagenes en data/review/<motivo>/<clase>/
                      Muestra PANEL DIVIDIDO: original (raw) + recorte actual.
                      El recorte se hace sobre la imagen ORIGINAL.

  --source rejected   Imagenes en data/rejected/<motivo>/<clase>/
                      Muestra la imagen original. Permite aceptar / rechazar.

  --source processed  Imagenes en data/processed/<clase>/
                      Auditoria de recortes ya aceptados: corregir o rechazar.

Controles comunes
-----------------
  A          Aceptar  → mueve a data/processed/<clase>/
  R          Rechazar → mueve a data/rejected/manual/<clase>/
  S / SPACE  Saltar (revisar despues)
  B          Volver a la imagen anterior
  Q / ESC    Salir y guardar progreso

Edicion (review y processed)
-----------------------------
  C          Modo recorte (click+arrastrar sobre imagen ORIGINAL en review)
  ENTER      Confirmar recorte
  H          Voltear horizontal
  V          Voltear vertical
  [          Rotar 90 izquierda
  ]          Rotar 90 derecha
  Z          Deshacer todos los cambios

Solo en review
--------------
  O          Mostrar / ocultar panel de imagen original

Uso rapido
----------
  python review_tool.py --source review
  python review_tool.py --source review    --subdir aspect_ratio
  python review_tool.py --source rejected  --subdir no_detection
  python review_tool.py --source processed
  python review_tool.py --source processed --subdir Turdus_fuscater
  python review_tool.py --source review    --list
  python review_tool.py --source review    --all    # re-revisar ya revisadas
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Colores BGR ──────────────────────────────────────────────────────────────
C_GREEN  = ( 50, 200,  50)
C_ORANGE = ( 30, 165, 255)
C_RED    = ( 50,  50, 240)
C_WHITE  = (255, 255, 255)
C_BLACK  = (  0,   0,   0)
C_GRAY   = (180, 180, 180)
C_YELLOW = (  0, 220, 220)
C_CYAN   = (200, 200,  30)
C_DARK   = ( 30,  30,  30)
C_DARKER = ( 20,  20,  20)

# ── Teclas ───────────────────────────────────────────────────────────────────
KEY_ACCEPT   = ord('a')
KEY_REJECT   = ord('r')
KEY_SKIP     = ord('s')
KEY_SPC      = 32
KEY_BACK     = ord('b')
KEY_QUIT     = ord('q')
KEY_ESC      = 27
KEY_CROP     = ord('c')
KEY_FLIP_H   = ord('h')
KEY_FLIP_V   = ord('v')
KEY_ROT_L    = ord('[')
KEY_ROT_R    = ord(']')
KEY_RESET    = ord('z')
KEY_CONFIRM  = 13
KEY_ORIG_TOG = ord('o')

HUD_H  = 58
CTRL_H = 28

IMG_EXTS = {".jpg", ".jpeg", ".png"}

SUBDIR_LABELS = {
    "low_confidence":        "Confianza baja",
    "small_area":            "Area pequena",
    "touches_edge":          "Toca borde",
    "aspect_ratio":          "Aspect ratio",
    "clipped":               "Clipeado",
    "final_too_small":       "Final muy pequeno",
    "no_detection":          "Sin deteccion YOLO",
    "bbox_empty_after_clip": "Bbox vacio tras clip",
    "load_error":            "Error de carga",
    "other":                 "Otro",
    "manual":                "Rechazado manual",
    "processed":             "Procesada (auditoria)",
}

SOURCE_META = {
    "review":    {"label": "REVIEW",    "color": C_ORANGE},
    "rejected":  {"label": "REJECTED",  "color": C_RED},
    "processed": {"label": "PROCESSED", "color": C_CYAN},
}

CONTROLS = {
    "review": (
        "[A] Aceptar  [R] Rechazar  [S] Saltar  [B] Volver  "
        "[C] Recortar ORIG  [H] FlipH  [V] FlipV  [[] RotL  []] RotR  "
        "[Z] Reset  [O] Toggle orig  [Q] Salir"
    ),
    "rejected": "[A] Aceptar  [R] Rechazar  [S] Saltar  [B] Volver  [Q] Salir",
    "processed": (
        "[A] Confirmar OK  [R] Rechazar  [S] Saltar  [B] Volver  "
        "[C] Re-recortar  [H] FlipH  [V] FlipV  [[] RotL  []] RotR  [Z] Reset  [Q] Salir"
    ),
    "crop": "[ENTER] Confirmar recorte   [Z / ESC] Cancelar",
}
BOX_SIZE = 224

# ═══════════════════════════════════════════════════════════════════════════════
#  Sesion
# ═══════════════════════════════════════════════════════════════════════════════

class ReviewSession:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"reviewed": {}, "stats": {"accepted": 0, "rejected": 0, "skipped": 0}}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def mark(self, key: str, decision: str):
        self.data["reviewed"][key] = {"decision": decision, "ts": time.time()}
        self.data["stats"][decision] = self.data["stats"].get(decision, 0) + 1
        self.save()

    def was_reviewed(self, key: str) -> bool:
        return key in self.data["reviewed"]

    def stats(self):
        return self.data["stats"]


# ═══════════════════════════════════════════════════════════════════════════════
#  DB
# ═══════════════════════════════════════════════════════════════════════════════

def db_update(db_path: Path, image_name: str, clase: str, status: str, reason: str = ""):
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE images SET status=?, rejection_reason=? WHERE image=? AND class=?",
            (status, reason, image_name, clase)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [DB] {e}")


def db_lookup(db_path: Path, image_name: str, clase: str):
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM images WHERE image=? AND class=?", (image_name, clase))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Colectar imagenes
# ═══════════════════════════════════════════════════════════════════════════════

def collect_review_or_rejected(base_dir: Path, subdirs_filter, class_filter=None):
    items = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdirs_filter and subdir.name not in subdirs_filter:
            continue
        for clase_dir in sorted(subdir.iterdir()):
            if not clase_dir.is_dir():
                continue
            if class_filter and clase_dir.name not in class_filter:
                continue
            for f in sorted(clase_dir.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    items.append({"path": f, "motivo": subdir.name, "clase": clase_dir.name})
    return items


def collect_processed(base_dir: Path, subdirs_filter):
    items = []
    for clase_dir in sorted(base_dir.iterdir()):
        if not clase_dir.is_dir():
            continue
        if subdirs_filter and clase_dir.name not in subdirs_filter:
            continue
        for f in sorted(clase_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                items.append({"path": f, "motivo": "processed", "clase": clase_dir.name})
    return items


def find_original(raw_dir: Path, clase: str, img_name: str):
    stem = Path(img_name).stem
    clase_dir = raw_dir / clase
    if not clase_dir.exists():
        return None
    for ext in IMG_EXTS:
        c = clase_dir / (stem + ext)
        if c.exists():
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Dibujo
# ═══════════════════════════════════════════════════════════════════════════════

def put_text_bg(img, text, xy, scale=0.50, thick=1, fg=C_WHITE, bg=C_DARK, alpha=0.85):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = xy
    ov = img.copy()
    cv2.rectangle(ov, (x-3, y-th-3), (x+tw+3, y+bl+2), bg, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)


def stamp_label(panel, text, color=C_YELLOW):
    out = panel.copy()
    put_text_bg(out, text, (6, 20), scale=0.46, fg=color)
    return out


def build_hud(canvas, info, mode="normal"):
    H, W = canvas.shape[:2]
    out = canvas.copy()

    cv2.rectangle(out, (0, 0), (W, HUD_H), C_DARK, -1)

    sm = SOURCE_META.get(info["source"], SOURCE_META["review"])
    put_text_bg(out, f"[{sm['label']}]", (8, 22), 0.60, 2, sm["color"], C_DARK, 1.0)
    put_text_bg(out, f"{info['idx']+1}/{info['total']}", (8, 48), 0.48, 1, C_GRAY, C_DARK, 1.0)

    motivo = SUBDIR_LABELS.get(info.get("motivo", ""), info.get("motivo", ""))
    put_text_bg(out, f"Motivo: {motivo}", (120, 22), 0.50, 1, C_YELLOW, C_DARK, 1.0)
    put_text_bg(out, f"Clase:  {info.get('clase','?')}", (120, 46), 0.48, 1, C_GRAY, C_DARK, 1.0)

    parts = [info.get("name", "")]
    if info.get("orig_w"):
        parts.append(f"orig {info['orig_w']}x{info['orig_h']}")
    parts.append(f"crop {info.get('w',0)}x{info.get('h',0)} px")
    if info.get("conf"):
        parts.append(f"conf={info['conf']:.2f}")
    meta_str = "   ".join(parts)
    put_text_bg(out, meta_str, (W - 480, 22), 0.43, 1, C_WHITE, C_DARK, 1.0)

    if info["source"] == "review":
        state = "ON" if info.get("show_orig", True) else "OFF"
        col   = C_GREEN if info.get("show_orig", True) else C_GRAY
        put_text_bg(out, f"[O] Orig: {state}", (W - 160, 46), 0.44, 1, col, C_DARK, 1.0)

    cv2.rectangle(out, (0, H-CTRL_H), (W, H), C_DARKER, -1)
    ctrl_key = "crop" if mode == "crop" else info["source"]
    ctrl_col = C_ORANGE if mode == "crop" else C_WHITE
    cv2.putText(out, CONTROLS.get(ctrl_key, ""), (6, H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, ctrl_col, 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Herramienta de recorte
# ═══════════════════════════════════════════════════════════════════════════════

class CropTool:
    def __init__(self):
        self.active = False
        self.px0 = self.py0 = self.px1 = self.py1 = 0
        self._px = self._py = 0
        self._scale = 1.0
        self._iw = self._ih = 0
        self.clicked = False

    def configure(self, panel_x, panel_y, scale, img_w, img_h):
        self._px    = panel_x
        self._py    = panel_y
        self._scale = scale
        self._iw    = img_w
        self._ih    = img_h

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.px0 = x
            self.py0 = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.px0 = x
            self.py0 = y
            self.clicked = True

    def _to_img(self, px, py):
        ix = (px - self._px) / self._scale
        iy = (py - self._py) / self._scale
        return max(0, min(int(ix), self._iw)), max(0, min(int(iy), self._ih))

    def get_img_rect(self):
        ix, iy = self._to_img(self.px0, self.py0)

        half = BOX_SIZE // 2

        x1 = max(0, ix - half)
        y1 = max(0, iy - half)
        x2 = min(self._iw, ix + half)
        y2 = min(self._ih, iy + half)

        # Ajuste si se sale del borde (mantener 224x224)
        if (x2 - x1) < BOX_SIZE:
            if x1 == 0:
                x2 = min(self._iw, BOX_SIZE)
            else:
                x1 = max(0, self._iw - BOX_SIZE)

        if (y2 - y1) < BOX_SIZE:
            if y1 == 0:
                y2 = min(self._ih, BOX_SIZE)
            else:
                y1 = max(0, self._ih - BOX_SIZE)

        return x1, y1, x2, y2

    def draw(self, canvas):
        half = int((BOX_SIZE * self._scale) / 2)

        x1 = int(self.px0 - half)
        y1 = int(self.py0 - half)
        x2 = int(self.px0 + half)
        y2 = int(self.py0 + half)

        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), C_ORANGE, -1)
        cv2.addWeighted(ov, 0.18, canvas, 0.82, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C_ORANGE, 2)

        put_text_bg(canvas, "224x224", (x1+4, y1+18), 0.46, 1, C_YELLOW, C_BLACK)


# ═══════════════════════════════════════════════════════════════════════════════
#  Canvas con panel dividido o unico
# ═══════════════════════════════════════════════════════════════════════════════

def make_canvas(win_w, win_h, crop_img, orig_img, show_orig, source):
    """
    Retorna (canvas_content, layout_dict).
    canvas_content: area entre HUD y barra de controles.
    layout_dict: describe el panel sobre el que opera CropTool.
    """
    avail_h = win_h - HUD_H - CTRL_H
    avail_w = win_w
    content = np.zeros((avail_h, avail_w, 3), dtype=np.uint8)
    layout  = {}

    use_split = show_orig and orig_img is not None

    if use_split:
        half = avail_w // 2 - 1
        div  = avail_w // 2
        content[:, div-1:div+1] = (60, 60, 60)   # divisor

        # Panel izquierdo: ORIGINAL
        oh, ow = orig_img.shape[:2]
        sc_o = min(half / ow, avail_h / oh)
        dw_o, dh_o = int(ow*sc_o), int(oh*sc_o)
        rsz_o = cv2.resize(orig_img, (dw_o, dh_o), interpolation=cv2.INTER_AREA)
        yo = (avail_h - dh_o) // 2
        xo = (half    - dw_o) // 2
        content[yo:yo+dh_o, xo:xo+dw_o] = rsz_o
        left = stamp_label(content[:avail_h, :div].copy(),
                           "ORIGINAL  [C] recortar aqui", C_YELLOW)
        content[:avail_h, :div] = left

        layout = {"panel_x": xo, "panel_y": yo, "scale": sc_o, "img_w": ow, "img_h": oh}

        # Panel derecho: RECORTE ACTUAL
        ch, cw = crop_img.shape[:2]
        sc_c = min(half / cw, avail_h / ch)
        dw_c, dh_c = int(cw*sc_c), int(ch*sc_c)
        rsz_c = cv2.resize(crop_img, (dw_c, dh_c), interpolation=cv2.INTER_AREA)
        yc = (avail_h - dh_c) // 2
        xc = div + 1 + (half - dw_c) // 2
        content[yc:yc+dh_c, xc:xc+dw_c] = rsz_c
        right = stamp_label(content[:avail_h, div:].copy(), "RECORTE ACTUAL", C_CYAN)
        content[:avail_h, div:] = right

    else:
        # Vista unica
        h, w = crop_img.shape[:2]
        sc = min(avail_w / w, avail_h / h, 1.0)
        dw, dh = int(w*sc), int(h*sc)
        rsz = cv2.resize(crop_img, (dw, dh), interpolation=cv2.INTER_AREA)
        y0 = (avail_h - dh) // 2
        x0 = (avail_w - dw) // 2
        content[y0:y0+dh, x0:x0+dw] = rsz
        layout = {"panel_x": x0, "panel_y": y0, "scale": sc, "img_w": w, "img_h": h}

    return content, layout


# ═══════════════════════════════════════════════════════════════════════════════
#  Bucle principal
# ═══════════════════════════════════════════════════════════════════════════════

def run_review(project_root, source, subdirs_filter, win_w=1280, win_h=800, skip_reviewed=True, class_filter=None):
    raw_dir       = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    rejected_dir  = project_root / "data" / "rejected"
    db_path       = project_root / "metadata" / "metadata.sqlite"
    session_file  = project_root / "metadata" / f"review_session_{source}.json"

    if source == "processed":
        source_dir = processed_dir
        all_items  = collect_processed(source_dir, subdirs_filter)
    else:
        source_dir = project_root / "data" / source
        if not source_dir.exists():
            print(f"[ERROR] No existe: {source_dir}")
            sys.exit(1)
        all_items = collect_review_or_rejected(source_dir, subdirs_filter, class_filter)

    session = ReviewSession(session_file)
    pending = [i for i in all_items if not session.was_reviewed(str(i["path"]))] \
              if skip_reviewed else list(all_items)

    print(f"\n{'='*64}")
    print(f"  Fuente    : {source_dir}")
    print(f"  Total     : {len(all_items)}")
    print(f"  Pendientes: {len(pending)}  (ya revisadas: {len(all_items)-len(pending)})")
    if source == "review":
        print("  Panel dividido: ORIGINAL (izquierda) | RECORTE ACTUAL (derecha)")
        print("  Tecla [C] hace el nuevo recorte sobre la imagen ORIGINAL")
    elif source == "processed":
        print("  Auditoria de imagenes ya aceptadas.")
    print(f"{'='*64}\n")

    if not pending:
        print("Sin imagenes pendientes.")
        _print_stats(session)
        return

    WIN = f"BirdLab Review [{source.upper()}]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w, win_h)

    idx      = 0
    history  = []
    crop_tool = CropTool()

    while 0 <= idx < len(pending):
        item      = pending[idx]
        img_path  = item["path"]
        motivo    = item["motivo"]
        clase     = item["clase"]

        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"  [WARN] No se pudo cargar {img_path}")
            idx += 1
            continue

        # Buscar original en raw/
        orig_bgr = None
        op = find_original(raw_dir, clase, img_path.name)
        if op:
            orig_bgr = cv2.imread(str(op))
        if orig_bgr is None:
            print(f"  [INFO] Original no encontrada para {img_path.name} "
                    f"(se mostrara solo el recorte)")

        meta = db_lookup(db_path, img_path.name, clase) or {}
        conf = meta.get("confidence")

        current_img = raw_bgr.copy()
        show_orig   = True
        crop_mode   = False
        crop_tool   = CropTool()
        decision    = None

        ih, iw = current_img.shape[:2]
        oh = ow = None
        if orig_bgr is not None:
            oh, ow = orig_bgr.shape[:2]

        info = {
            "source":    source,
            "name":      img_path.name,
            "clase":     clase,
            "motivo":    motivo,
            "idx":       idx,
            "total":     len(pending),
            "w": iw, "h": ih,
            "orig_w": ow, "orig_h": oh,
            "conf":      conf,
            "show_orig": show_orig,
        }

        while True:
            info["show_orig"] = show_orig
            info["w"] = current_img.shape[1]
            info["h"] = current_img.shape[0]

            content, layout = make_canvas(
                win_w, win_h,
                crop_img  = current_img,
                orig_img  = orig_bgr,
                show_orig = show_orig,
                source    = source,
            )

            full = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            full[HUD_H: win_h-CTRL_H] = content

            mode_str = "crop" if crop_mode else "normal"
            full = build_hud(full, info, mode_str)

            if crop_mode:
                # El panel de recorte es siempre el IZQUIERDO (original) en review,
                # o el unico panel en processed/rejected.
                real_py = HUD_H + layout["panel_y"]
                crop_tool.configure(
                    panel_x = layout["panel_x"],
                    panel_y = real_py,
                    scale   = layout["scale"],
                    img_w   = layout["img_w"],
                    img_h   = layout["img_h"],
                )
                cv2.setMouseCallback(WIN, crop_tool.mouse_cb)
                crop_tool.draw(full)
            else:
                cv2.setMouseCallback(WIN, lambda *a: None)

            cv2.imshow(WIN, full)
            key = cv2.waitKey(30) & 0xFF

            # ── Modo recorte ─────────────────────────────────────────────────
            if crop_mode:
                if key == KEY_CONFIRM:
                    x1, y1, x2, y2 = crop_tool.get_img_rect()
                    if (x2-x1) > 10 and (y2-y1) > 10:
                        # En review + orig disponible: recortar sobre ORIGINAL
                        src = (orig_bgr if (source in ("review", "processed") and orig_bgr is not None)
                               else current_img)
                        current_img = src[y1:y2, x1:x2].copy()
                    crop_mode = False
                elif key in (KEY_RESET, KEY_ESC):
                    crop_mode = False
                if crop_tool.clicked:
                    x1, y1, x2, y2 = crop_tool.get_img_rect()

                    src = (orig_bgr if (source in ("review", "processed") and orig_bgr is not None)
                        else current_img)

                    current_img = src[y1:y2, x1:x2].copy()

                    crop_tool.clicked = False
                    crop_mode = False
                continue

            # ── Edicion ──────────────────────────────────────────────────────
            if source in ("review", "processed"):
                if key == KEY_CROP:
                    crop_mode = True
                    crop_tool = CropTool()
                    continue
                elif key == KEY_FLIP_H:
                    current_img = cv2.flip(current_img, 1)
                elif key == KEY_FLIP_V:
                    current_img = cv2.flip(current_img, 0)
                elif key == KEY_ROT_L:
                    current_img = cv2.rotate(current_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif key == KEY_ROT_R:
                    current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)
                elif key == KEY_RESET:
                    current_img = raw_bgr.copy()

            if source == "review" and key == KEY_ORIG_TOG:
                show_orig = not show_orig
                continue

            # ── Navegacion / decision ────────────────────────────────────────
            if   key == KEY_ACCEPT:             decision = "accepted";  break
            elif key == KEY_REJECT:             decision = "rejected";  break
            elif key in (KEY_SKIP, KEY_SPC):    decision = "skipped";   break
            elif key == KEY_BACK:
                if history:
                    idx = history.pop()
                break
            elif key in (KEY_QUIT, KEY_ESC):
                print("\n[INFO] Saliendo.")
                cv2.destroyAllWindows()
                _print_stats(session)
                return

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Ventana cerrada.")
                _print_stats(session)
                return

        # ── Procesar decision ────────────────────────────────────────────────
        if decision is None:
            continue

        if decision == "accepted":
            dest = processed_dir / clase / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), current_img)
            # Solo borrar si el archivo origen es distinto al destino
            if img_path.resolve() != dest.resolve():
                img_path.unlink(missing_ok=True)
            db_update(db_path, img_path.name, clase, "accepted", "")
            print(f"  [✓] Aceptada  → {dest.relative_to(project_root)}")

        elif decision == "rejected":
            dest = rejected_dir / "manual" / clase / img_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), raw_bgr)   # original sin edicion al rechazar
            if img_path.resolve() != dest.resolve():
                img_path.unlink(missing_ok=True)
            db_update(db_path, img_path.name, clase, "rejected", "manual_review")
            print(f"  [✗] Rechazada → {dest.relative_to(project_root)}")

        elif decision == "skipped":
            print(f"  [~] Saltada:   {img_path.name}")

        session.mark(str(img_path), decision)
        history.append(idx)
        idx += 1

    cv2.destroyAllWindows()
    print(f"\n{'='*50}")
    print("  Revision completada")
    _print_stats(session)
    print(f"{'='*50}")


def _print_stats(session: ReviewSession):
    s = session.stats()
    print(f"  Aceptadas : {s.get('accepted', 0)}")
    print(f"  Rechazadas: {s.get('rejected', 0)}")
    print(f"  Saltadas  : {s.get('skipped',  0)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="W&F BirdLab -- review_tool v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project", type=Path, default=None,
        help="Raiz del proyecto (default: dos niveles arriba del script)"
    )
    parser.add_argument(
        "--source", choices=["review", "rejected", "processed"], default="review",
        help="Fuente de imagenes (default: review)"
    )
    parser.add_argument(
        "--subdir", nargs="*", default=None,
        help=(
            "review/rejected: motivo(s) a filtrar. "
            "processed: clase(s) a filtrar."
        )
    )
    parser.add_argument(
        "--class", dest="class_filter", nargs="*", default=None,
        help="Filtrar por clase(s), ej: piaya-cayana"
    )
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--all", action="store_true",
        help="Re-revisar imagenes ya marcadas en sesiones anteriores"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Listar imagenes pendientes sin abrir ventana"
    )

    args = parser.parse_args()

    project_root = (args.project.resolve() if args.project
                    else Path(__file__).resolve().parent.parent)

    if not project_root.exists():
        print(f"[ERROR] Proyecto no encontrado: {project_root}")
        sys.exit(1)

    if args.list:
        if args.source == "processed":
            base  = project_root / "data" / "processed"
            items = collect_processed(base, args.subdir)
        else:
            base  = project_root / "data" / args.source
            items = collect_review_or_rejected(base, args.subdir)

        print(f"\nImagenes en {base}  ({len(items)} total)\n")
        groups: dict = {}
        for it in items:
            groups.setdefault(it["motivo"], {}).setdefault(it["clase"], 0)
            groups[it["motivo"]][it["clase"]] += 1
        for motivo, clases in sorted(groups.items()):
            print(f"  {SUBDIR_LABELS.get(motivo, motivo)} ({motivo}/): "
                  f"{sum(clases.values())} imgs")
            for cls, n in sorted(clases.items()):
                print(f"    {cls}: {n}")
        return

    run_review(
        project_root   = project_root,
        source         = args.source,
        subdirs_filter = args.subdir,
        win_w          = args.width,
        win_h          = args.height,
        skip_reviewed  = not args.all,
        class_filter   = args.class_filter,
    )


if __name__ == "__main__":
    main()
"""
src/explainability.py
─────────────────────
Técnicas de explicabilidad visual para el clasificador de aves W&F BirdLab.

Implementa:
  - Grad-CAM  : muestra QUÉ REGIÓN de la imagen activó la predicción.
  - LIME      : muestra QUÉ SEGMENTOS de la imagen son más relevantes.

Diseño:
  - Todas las funciones son stateless: reciben modelo, imagen y devuelven arrays.
  - Compatibles con MobileNetV3-Small (backbone congelado o con dropout).
  - Sin dependencia del sistema de rutas del proyecto: usables desde cualquier
    notebook o script con solo pasar el modelo y la imagen.

Dependencias:
  pip install grad-cam lime scikit-image
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── Grad-CAM ─────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── LIME ──────────────────────────────────────────────────────────
from lime import lime_image
from skimage.segmentation import mark_boundaries

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════
#  Normalización ImageNet (debe coincidir con EVAL_TRANSFORM)
# ══════════════════════════════════════════════════════════════════

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize(img_np: np.ndarray) -> np.ndarray:
    """img_np: float32 [0,1] (H,W,3) → normalizado con media/std ImageNet."""
    return (img_np - _MEAN) / _STD


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """(H,W,3) float32 [0,1] → tensor (1,3,H,W)."""
    normed = _normalize(img_np)
    return torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float()


def load_image(
    path: Union[str, Path],
    size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga y redimensiona una imagen.

    Returns:
        img_float : np.ndarray float32 [0,1] (H,W,3) — para Grad-CAM y LIME
        img_uint8 : np.ndarray uint8  [0,255] (H,W,3) — para LIME predict_fn
    """
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    img_uint8  = np.array(img, dtype=np.uint8)
    img_float  = img_uint8.astype(np.float32) / 255.0
    return img_float, img_uint8


# ══════════════════════════════════════════════════════════════════
#  Grad-CAM
# ══════════════════════════════════════════════════════════════════

def get_gradcam_target_layer(model: nn.Module) -> list:
    """
    Devuelve la capa objetivo para Grad-CAM según la arquitectura.

    MobileNetV3 (Small y Large): ultima capa de features (Conv2dNormActivation).
    Si no se reconoce la arquitectura, usa la última capa de 'features'.
    """
    if hasattr(model, "features"):
        return [model.features[-1]]
    # Fallback genérico: buscar el ultimo bloque Conv/BN
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return [convs[-1]]
    raise ValueError(
        "No se pudo determinar la capa objetivo para Grad-CAM. "
        "Pasa target_layer manualmente."
    )


def explain_gradcam(
    model: nn.Module,
    img_float: np.ndarray,
    target_class: Optional[int] = None,
    target_layer: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, int, float]:
    """
    Aplica Grad-CAM sobre una imagen y devuelve el mapa de calor superpuesto.

    Args:
        model        : modelo PyTorch en eval(), con backbone cargado.
        img_float    : np.ndarray float32 [0,1] (H,W,3).
        target_class : índice de clase objetivo. None → usa la predicción del modelo.
        target_layer : lista con la(s) capa(s) objetivo. None → auto-detect.
        device       : torch.device. None → usa el dispositivo del modelo.

    Returns:
        cam_image    : np.ndarray float32 [0,1] (H,W,3) — imagen + heatmap RGB.
        pred_class   : índice de la clase predicha.
        pred_conf    : confianza (softmax) de la clase predicha.
    """
    if device is None:
        device = next(model.parameters()).device

    if target_layer is None:
        target_layer = get_gradcam_target_layer(model)

    model.eval()
    input_tensor = _to_tensor(img_float).to(device)

    # Predicción para saber qué clase usar si target_class es None
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    pred_conf  = float(probs[pred_class].item())

    if target_class is None:
        target_class = pred_class

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # grayscale_cam: (1, H, W) → (H, W)
    heatmap   = grayscale_cam[0]
    cam_image = show_cam_on_image(img_float, heatmap, use_rgb=True)

    return cam_image, pred_class, pred_conf


# ══════════════════════════════════════════════════════════════════
#  LIME
# ══════════════════════════════════════════════════════════════════

def _make_predict_fn(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
):
    """
    Crea la función de predicción que espera LIME.

    LIME pasa un array uint8 (N, H, W, 3). La función debe devolver (N, C) float.
    """
    model.eval()

    def predict_fn(images: np.ndarray) -> np.ndarray:
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            # uint8 [0,255] → float [0,1] → normalizar → tensor
            batch_f  = batch.astype(np.float32) / 255.0
            batch_n  = (_normalize(batch_f) if batch_f.ndim == 4
                        else _normalize(batch_f[np.newaxis]))[:]
            # (N,H,W,3) → (N,3,H,W)
            tensors  = torch.from_numpy(
                batch_n.transpose(0, 3, 1, 2)
            ).float().to(device)
            with torch.no_grad():
                logits = model(tensors)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    return predict_fn


def explain_lime(
    model: nn.Module,
    img_uint8: np.ndarray,
    target_class: Optional[int] = None,
    num_samples: int = 1000,
    num_features: int = 8,
    positive_only: bool = True,
    hide_rest: bool = False,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Aplica LIME sobre una imagen y devuelve la imagen segmentada con límites.

    Args:
        model         : modelo PyTorch en eval().
        img_uint8     : np.ndarray uint8 [0,255] (H,W,3).
        target_class  : clase a explicar. None → predicción del modelo.
        num_samples   : perturbaciones que genera LIME (más = más preciso, más lento).
        num_features  : segmentos a mostrar como más relevantes.
        positive_only : solo mostrar superpíxeles que APOYAN la predicción.
        hide_rest     : oscurecer los superpíxeles no relevantes.
        random_seed   : semilla para reproducibilidad.
        device        : torch.device.

    Returns:
        lime_image    : np.ndarray float64 (H,W,3) — imagen con límites de segmentos.
        mask          : np.ndarray int (H,W) — máscara de superpíxeles relevantes.
        pred_class    : clase predicha.
        pred_conf     : confianza de la clase predicha.
    """
    if device is None:
        device = next(model.parameters()).device

    predict_fn = _make_predict_fn(model, device)

    # Predicción sobre la imagen original para obtener clase y confianza
    probs      = predict_fn(img_uint8[np.newaxis])[0]
    pred_class = int(probs.argmax())
    pred_conf  = float(probs[pred_class])

    if target_class is None:
        target_class = pred_class

    explainer   = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img_uint8,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        random_seed=random_seed,
    )

    img_lime, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=hide_rest,
    )

    # Superponer bordes de segmentación
    lime_result = mark_boundaries(
        img_lime.astype(np.float64) / 255.0,
        mask,
        color=(1, 0.5, 0),   # naranja para los bordes
        mode="thick",
    )

    return lime_result, mask, pred_class, pred_conf


# ══════════════════════════════════════════════════════════════════
#  Función de alto nivel: explicar N imágenes de clases distintas
# ══════════════════════════════════════════════════════════════════

def explain_samples(
    model: nn.Module,
    samples: list[dict],
    classes: list[str],
    size: int = 224,
    lime_num_samples: int = 1000,
    lime_num_features: int = 8,
    device: Optional[torch.device] = None,
) -> list[dict]:
    """
    Explica una lista de imágenes con Grad-CAM y LIME.

    Args:
        model      : modelo PyTorch en eval() con pesos cargados.
        samples    : lista de dicts con keys:
                       "path"  → ruta a la imagen
                       "class" → nombre de la clase (para el título)
        classes    : lista global de clases del modelo (class_to_idx invertido).
        size       : tamaño al que redimensionar (debe coincidir con el entrenamiento).
        lime_num_samples : perturbaciones LIME por imagen.
        lime_num_features: superpíxeles a destacar en LIME.
        device     : torch.device. None → auto.

    Returns:
        Lista de dicts con:
          "path", "true_class", "pred_class", "pred_conf",
          "img_float", "img_uint8",
          "gradcam_img",
          "lime_img", "lime_mask"
    """
    if device is None:
        device = next(model.parameters()).device

    results = []
    for sample in samples:
        img_float, img_uint8 = load_image(sample["path"], size=size)

        # ── Grad-CAM ─────────────────────────────────────────────
        cam_img, pred_class, pred_conf = explain_gradcam(
            model, img_float, device=device
        )

        # ── LIME ─────────────────────────────────────────────────
        lime_img, lime_mask, _, _ = explain_lime(
            model, img_uint8,
            num_samples=lime_num_samples,
            num_features=lime_num_features,
            device=device,
        )

        results.append({
            "path":        Path(sample["path"]),
            "true_class":  sample.get("class", "?"),
            "pred_class":  classes[pred_class],
            "pred_conf":   pred_conf,
            "img_float":   img_float,
            "img_uint8":   img_uint8,
            "gradcam_img": cam_img,
            "lime_img":    lime_img,
            "lime_mask":   lime_mask,
        })

    return results

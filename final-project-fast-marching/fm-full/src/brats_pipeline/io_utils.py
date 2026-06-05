"""
io_utils.py
===========
Entrada/salida de volúmenes y localización de archivos por caso.

Usamos **SimpleITK** como motor de E/S del pipeline de limpieza porque:
  * N4 (corrección de campo de sesgo) solo existe en SimpleITK/ITK, y
  * `sitk.ReadImage`/`WriteImage` preservan automáticamente spacing, origen,
    dirección y affine. Así los volúmenes limpios quedan geométricamente
    idénticos a los crudos y nadie del proyecto hereda un desfase de ejes.

Convención de nombres de BraTS-GLI (igual a la del EDA):
    <DATASET_DIR>/<case_id>/<case_id>-t1n.nii.gz   (y -t1c, -t2w, -t2f, -seg)
"""
from __future__ import annotations
import glob
import os
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk

from . import config


# --------------------------------------------------------------------------- #
# Localización de archivos
# --------------------------------------------------------------------------- #
def ruta_modalidad(case_id: str, mod: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Devuelve la ruta al NIfTI de `mod` para `case_id`, o None si no existe."""
    base_dir = base_dir or config.DATASET_DIR
    patron = os.path.join(base_dir, case_id, f"*-{mod}.nii*")
    hits = sorted(glob.glob(patron))
    return hits[0] if hits else None


def rutas_caso(case_id: str, base_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Diccionario {mod: ruta} para todas las modalidades + seg de un caso."""
    return {m: ruta_modalidad(case_id, m, base_dir) for m in config.SUFIJOS_SALIDA}


# --------------------------------------------------------------------------- #
# Carga / guardado
# --------------------------------------------------------------------------- #
def leer_imagen(path: str) -> sitk.Image:
    """Lee un NIfTI como imagen de SimpleITK (geometría preservada)."""
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return sitk.ReadImage(path, sitk.sitkFloat32)


def guardar_imagen(img: sitk.Image, path: str) -> None:
    """Escribe una imagen de SimpleITK a disco creando la carpeta destino."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(img, path, useCompression=True)


def a_numpy(img: sitk.Image) -> np.ndarray:
    """Imagen SITK -> ndarray. OJO: el orden de ejes queda (z, y, x)."""
    return sitk.GetArrayFromImage(img)


def desde_numpy(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    """ndarray (z, y, x) -> imagen SITK copiando la geometría de `ref`."""
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def mascara_cerebro(img: sitk.Image, umbral: float = 0.0) -> sitk.Image:
    """
    Máscara de cerebro. Los volúmenes de BraTS vienen *skull-stripped*: el fondo
    es exactamente 0, así que `img > 0` es una máscara de cerebro fiable y barata.
    Devuelve una imagen uint8 (1 = cerebro, 0 = fondo).
    """
    return sitk.Cast(img > umbral, sitk.sitkUInt8)


# --------------------------------------------------------------------------- #
# Guardia de geometría (defensiva: otras personas usan estas salidas)
# --------------------------------------------------------------------------- #
def verificar_geometria(img: sitk.Image, case_id: str,
                        spacing_obj=config.SPACING_OBJETIVO, tol: float = 1e-3) -> bool:
    """
    Comprueba que la imagen esté en el spacing objetivo (1mm iso). Devuelve True/False
    y NO lanza excepción: solo avisa, porque el EDA confirmó que todo el GLI ya viene
    isotrópico a 1mm. Sirve de red de seguridad si entra un caso atípico.
    """
    sp = np.array(img.GetSpacing(), dtype=float)
    ok = np.allclose(sp, np.array(spacing_obj), atol=tol)
    if not ok:
        print(f"  [AVISO] {case_id}: spacing {tuple(np.round(sp,3))} != {spacing_obj}. "
              f"Geometría confirmó 1mm iso en el EDA; revisar este caso antes de seguir.")
    return ok

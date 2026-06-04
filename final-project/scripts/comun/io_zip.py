"""
io_zip.py
=========
Entrada/salida de volúmenes leyendo los .nii.gz DIRECTAMENTE desde los ZIP de BraTS,
sin descomprimir el archivo completo. Solo se extrae a un temporal el caso+modalidad
que se necesita en cada momento (patrón heredado de nuestro notebook de Segmentación).

Motor de imagen: **SimpleITK** (sitk). `ReadImage`/`WriteImage` preservan spacing, origen
y dirección, así que los volúmenes procesados quedan geométricamente idénticos a los crudos.

Las funciones de máscara/geometría (`mascara_cerebro`, `a_numpy`, `desde_numpy`,
`verificar_geometria`) están copiadas TAL CUAL del io_utils.py de Santiago; lo único que
reescribimos respecto a él es la **localización**: aquí se lee del ZIP, no de carpetas.

Convención de nombres BraTS-GLI:  <case_id>/<case_id>-{t1n,t1c,t2w,t2f,seg}.nii.gz
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk

from . import constantes as C


# --------------------------------------------------------------------------- #
# Descubrimiento de casos dentro del ZIP
# --------------------------------------------------------------------------- #
_RE_CASE = re.compile(r"(BraTS-GLI-\d+-\d+)")


def listar_casos(zip_path: Path = C.ZIP_TRAINING,
                 max_casos: Optional[int] = None) -> list[str]:
    """
    Lista los case_id presentes en el ZIP (ordenados), sin descomprimir.
    Un caso se detecta por la presencia de su archivo -t1c.nii.gz.

    Parameters
    ----------
    zip_path : Path al ZIP de BraTS.
    max_casos : si se da, recorta a los primeros `max_casos` casos.
    """
    casos: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for nombre in zf.namelist():
            if nombre.endswith("-t1c.nii.gz"):
                m = _RE_CASE.search(nombre)
                if m:
                    casos.add(m.group(1))
    ordenados = sorted(casos)
    if max_casos is not None:
        ordenados = ordenados[:max_casos]
    return ordenados


def tiene_modalidad(zip_path: Path, case_id: str, mod: str) -> bool:
    """True si el ZIP contiene <case_id>-<mod>.nii.gz."""
    suf = f"{case_id}-{mod}.nii.gz"
    with zipfile.ZipFile(zip_path) as zf:
        return any(n.endswith(suf) for n in zf.namelist())


# --------------------------------------------------------------------------- #
# Extracción puntual desde el ZIP
# --------------------------------------------------------------------------- #
def extraer_nii(case_id: str, mod: str,
                zip_path: Path = C.ZIP_TRAINING,
                tmp_dir: Path = C.TMP_DIR) -> Optional[Path]:
    """
    Extrae SOLO <case_id>-<mod>.nii.gz a `tmp_dir` y devuelve su ruta (o None si no está).
    Cachea: si el archivo ya fue extraído, no lo vuelve a copiar.
    """
    suf = f"{case_id}-{mod}.nii.gz"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    destino = tmp_dir / suf
    if destino.exists():
        return destino
    with zipfile.ZipFile(zip_path) as zf:
        miembro = next((n for n in zf.namelist() if n.endswith(suf)), None)
        if miembro is None:
            return None
        with zf.open(miembro) as origen, open(destino, "wb") as salida:
            shutil.copyfileobj(origen, salida)
    return destino


# --------------------------------------------------------------------------- #
# Lectura / guardado SimpleITK
# --------------------------------------------------------------------------- #
def leer_sitk(case_id: str, mod: str,
              zip_path: Path = C.ZIP_TRAINING,
              tmp_dir: Path = C.TMP_DIR,
              como_float: bool = True) -> Optional[sitk.Image]:
    """
    Lee una modalidad como imagen SimpleITK (geometría preservada). None si no existe.
    `como_float=True` la carga en float32 (recomendado para procesamiento);
    para la segmentación (-seg) usar `como_float=False` y conservar etiquetas enteras.
    """
    ruta = extraer_nii(case_id, mod, zip_path, tmp_dir)
    if ruta is None:
        return None
    if como_float:
        return sitk.ReadImage(str(ruta), sitk.sitkFloat32)
    return sitk.ReadImage(str(ruta))


def leer_np(case_id: str, mod: str,
            zip_path: Path = C.ZIP_TRAINING,
            tmp_dir: Path = C.TMP_DIR) -> Optional[np.ndarray]:
    """Atajo: modalidad -> ndarray (z, y, x). None si no existe."""
    img = leer_sitk(case_id, mod, zip_path, tmp_dir)
    return None if img is None else a_numpy(img)


def leer_seg_np(case_id: str,
                zip_path: Path = C.ZIP_TRAINING,
                tmp_dir: Path = C.TMP_DIR) -> Optional[np.ndarray]:
    """Lee la segmentación GT como ndarray int16 (z, y, x), preservando etiquetas."""
    img = leer_sitk(case_id, "seg", zip_path, tmp_dir, como_float=False)
    return None if img is None else a_numpy(img).astype(np.int16)


def guardar_sitk(img: sitk.Image, path: Path) -> Path:
    """Escribe una imagen SimpleITK a disco (crea carpeta, compresión activada)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)
    return path


# --------------------------------------------------------------------------- #
# Conversiones y máscara de cerebro (copiadas de io_utils.py de Santiago)
# --------------------------------------------------------------------------- #
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
    Máscara de cerebro. BraTS viene *skull-stripped*: el fondo es exactamente 0, así
    que `img > 0` es una máscara fiable y barata. Devuelve uint8 (1=cerebro, 0=fondo).
    """
    return sitk.Cast(img > umbral, sitk.sitkUInt8)


# --------------------------------------------------------------------------- #
# Guardia de geometría (copiada de Santiago; defensiva, no lanza excepción)
# --------------------------------------------------------------------------- #
def verificar_geometria(img: sitk.Image, case_id: str,
                        spacing_obj: Iterable[float] = C.SPACING_OBJETIVO,
                        tol: float = 1e-3) -> bool:
    """
    Comprueba que la imagen esté en el spacing objetivo (1mm iso). Devuelve True/False
    y NO lanza excepción: solo avisa. El EDA confirmó que el GLI ya viene isotrópico a
    1mm; sirve de red de seguridad ante un caso atípico.
    """
    sp = np.array(img.GetSpacing(), dtype=float)
    ok = bool(np.allclose(sp, np.array(list(spacing_obj)), atol=tol))
    if not ok:
        print(f"  [AVISO] {case_id}: spacing {tuple(np.round(sp, 3))} != {tuple(spacing_obj)}. "
              f"Geometría confirmó 1mm iso en el EDA; revisar este caso.")
    return ok

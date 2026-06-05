"""
clean_pipeline.py
=================
Orquesta la limpieza de UN caso y deja la salida lista para el resto del equipo:

    final-project/output/limpieza/<case_id>/
        <case_id>-t1n.nii.gz   (denoise -> N4 -> normalización)
        <case_id>-t1c.nii.gz
        <case_id>-t2w.nii.gz
        <case_id>-t2f.nii.gz
        <case_id>-seg.nii.gz   (COPIADA tal cual; la GT no se filtra ni se normaliza)

Orden de operaciones (el del enunciado): **denoise -> N4 -> normalización**.
  1. Denoise primero reduce el ruido que, si no, N4 podría interpretar como
     estructura de bajo nivel.
  2. N4 corrige el sesgo multiplicativo.
  3. Normalización al final fija la escala entre casos (Sección D del EDA).
(Si prefieres N4->denoise, es defendible; basta reordenar las llamadas abajo.)

La segmentación (-seg) se COPIA byte a byte con shutil para preservar exactamente
geometría y etiquetas: no debe pasar por ningún filtro.
"""
from __future__ import annotations
import os
import shutil
from typing import Dict, Optional

import SimpleITK as sitk

from . import config, io_utils, denoise, bias_field, normalize


def limpiar_caso(case_id: str,
                 base_dir: Optional[str] = None,
                 out_root: Optional[str] = None,
                 mysize_wiener: int = 3,
                 n4_shrink: int = 4,
                 esquema_norm: str = "zscore",
                 verbose: bool = True) -> Dict[str, str]:
    """
    Limpia un caso y escribe los 5 archivos de salida.

    Returns
    -------
    dict {mod: ruta_salida} con lo efectivamente escrito.
    """
    base_dir = base_dir or config.DATASET_DIR
    out_root = out_root or config.OUT_LIMPIEZA
    rutas = io_utils.rutas_caso(case_id, base_dir)
    destino = os.path.join(out_root, case_id)
    os.makedirs(destino, exist_ok=True)
    escritos: Dict[str, str] = {}

    if verbose:
        print(f"[{case_id}] limpiando...")

    # --- 4 modalidades: denoise -> N4 -> normalización -----------------------
    for mod in config.MODALITIES_IMG:
        src = rutas.get(mod)
        if not src:
            print(f"  [AVISO] falta modalidad {mod}; se omite.")
            continue

        img = io_utils.leer_imagen(src)
        io_utils.verificar_geometria(img, case_id)           # red de seguridad
        mascara = io_utils.mascara_cerebro(img)              # x>0 (skull-stripped)

        img = denoise.denoise_imagen(img, mysize=mysize_wiener)          # 1) ruido
        img = bias_field.corregir_n4(img, mascara=mascara, shrink_factor=n4_shrink)  # 2) N4
        img = normalize.normalizar_imagen(img, esquema=esquema_norm)     # 3) normalización

        dst = os.path.join(destino, f"{case_id}-{mod}.nii.gz")
        io_utils.guardar_imagen(img, dst)
        escritos[mod] = dst
        if verbose:
            print(f"  ok {mod} -> {os.path.relpath(dst, out_root)}")

    # --- segmentación: copia tal cual ---------------------------------------
    seg_src = rutas.get("seg")
    if seg_src and os.path.exists(seg_src):
        dst_seg = os.path.join(destino, f"{case_id}-seg.nii.gz")
        shutil.copyfile(seg_src, dst_seg)
        escritos["seg"] = dst_seg
        if verbose:
            print(f"  ok seg (copiada) -> {os.path.relpath(dst_seg, out_root)}")
    else:
        print(f"  [AVISO] no se encontró seg para {case_id}.")

    return escritos


def limpiar_lote(case_ids, **kwargs) -> Dict[str, Dict[str, str]]:
    """Limpia una lista de casos. Devuelve {case_id: {mod: ruta}}."""
    resultados = {}
    for cid in case_ids:
        resultados[cid] = limpiar_caso(cid, **kwargs)
    print(f"\nListo: {len(resultados)} casos limpios en {kwargs.get('out_root', config.OUT_LIMPIEZA)}")
    return resultados

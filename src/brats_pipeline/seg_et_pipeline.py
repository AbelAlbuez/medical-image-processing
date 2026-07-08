"""
seg_et_pipeline.py
==================
Segmentación del Enhancing Tumor (ET) en BraTS 2024 GLI.
Implementa 4 métodos clásicos + 1 semi-automático con semilla manual.

Modalidad principal: T1c (gadolinio realza el ET).
Señal auxiliar: mapa T1c−T1n (normalización conjunta) para aislar el realce.

Métodos:
  1. otsu_T1c       — Otsu multinivel n=3 sobre T1c limpio
  2. gmm_T1c        — GMM 3 componentes sobre T1c limpio
  3. sustraccion    — umbralización del mapa T1c−T1n (norm. conjunta)
  4. gmm_2d         — GMM 4 comp. sobre [T1c, mapa_dif] con filtro de compacidad
  5. semilla        — Region growing desde semilla manual en el mapa_dif
                      (semi-automático; requiere coordenadas de semilla)
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import random
import time
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from sklearn.mixture import GaussianMixture

import SimpleITK as sitk
from . import io_utils, config
from .seg_metrics import dice, jaccard


# ================================================================== #
# Utilidades                                                          #
# ================================================================== #

def _cerebro(arr: np.ndarray) -> np.ndarray:
    return arr != 0


def _normalizar_conjunto(a1: np.ndarray, a2: np.ndarray,
                          mask: np.ndarray, p: float = 99.5):
    """Normaliza dos volúmenes con el MISMO factor (máximo conjunto).
    Preserva la relación de escala entre T1c y T1n — clave para la sustracción."""
    vmax = max(float(np.percentile(a1[mask], p)),
               float(np.percentile(a2[mask], p)))
    if vmax <= 0:
        vmax = 1.0
    return (np.clip(a1 / vmax, 0, 1).astype(np.float32),
            np.clip(a2 / vmax, 0, 1).astype(np.float32))


def _mapa_diferencia(t1c_raw: np.ndarray, t1n_raw: np.ndarray,
                      sigma: float = 0.5) -> np.ndarray:
    """Calcula mapa T1c−T1n con normalización conjunta y suavizado."""
    mask = t1c_raw > 0
    t1c_n, t1n_n = _normalizar_conjunto(t1c_raw, t1n_raw, mask)
    mapa = t1c_n - t1n_n
    if sigma > 0:
        mapa = ndimage.gaussian_filter(mapa, sigma=sigma)
    return mapa


def _morfologia(mask: np.ndarray, erosion: int = 1,
                dilatacion: int = 1, keep_largest: bool = True) -> np.ndarray:
    struct = ndimage.generate_binary_structure(3, 1)
    if erosion > 0:
        mask = ndimage.binary_erosion(mask, structure=struct, iterations=erosion)
    if dilatacion > 0:
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=dilatacion)
    if keep_largest and mask.any():
        labeled, n = ndimage.label(mask, structure=struct)
        if n > 1:
            sizes = ndimage.sum(mask, labeled, range(1, n + 1))
            mask = labeled == (int(np.argmax(sizes)) + 1)
    return mask.astype(np.uint8)


# ================================================================== #
# Método 1: Otsu sobre T1c                                           #
# ================================================================== #

def metodo_otsu(arr_t1c: np.ndarray, clases: int = 3) -> np.ndarray:
    cerebro = _cerebro(arr_t1c)
    vals = arr_t1c[cerebro]
    if vals.size < clases or np.unique(vals).size < clases:
        return np.zeros_like(arr_t1c, dtype=np.uint8)
    umbrales = threshold_multiotsu(vals, classes=clases)
    etiquetas = np.digitize(arr_t1c, bins=umbrales)
    pred = ((etiquetas == clases - 1) & cerebro).astype(np.uint8)
    return _morfologia(pred, 1, 1, keep_largest=True)


# ================================================================== #
# Método 2: GMM 1D sobre T1c                                         #
# ================================================================== #

def metodo_gmm(arr_t1c: np.ndarray, n_comp: int = 3,
               seed: int = 42) -> np.ndarray:
    cerebro = _cerebro(arr_t1c)
    X = arr_t1c[cerebro].reshape(-1, 1).astype(np.float32)
    if X.shape[0] < n_comp:
        return np.zeros_like(arr_t1c, dtype=np.uint8)
    gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                          random_state=seed)
    labels = gmm.fit_predict(X)
    medias = [X[labels == c].mean() if (labels == c).any() else -np.inf
              for c in range(n_comp)]
    cluster_et = int(np.argmax(medias))
    pred = np.zeros_like(arr_t1c, dtype=np.uint8)
    pred[cerebro] = (labels == cluster_et).astype(np.uint8)
    return _morfologia(pred, 1, 1, keep_largest=True)


# ================================================================== #
# Método 3: Sustracción T1c−T1n                                      #
# ================================================================== #

def metodo_sustraccion(arr_t1c_raw: np.ndarray, arr_t1n_raw: np.ndarray,
                        auto_pct: float = 90.0, sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    cerebro = arr_t1c_raw > 0
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma)
    vals_pos = mapa[cerebro & (mapa > 0)]
    umbral = float(np.percentile(vals_pos, auto_pct)) if len(vals_pos) > 0 else 0.0
    pred = ((mapa > umbral) & cerebro).astype(np.uint8)
    pred = _morfologia(pred, 1, 1, keep_largest=False)
    return pred, mapa, umbral


# ================================================================== #
# Método 4: GMM 2D sobre [T1c, mapa_dif]                            #
# ================================================================== #

def metodo_gmm_2d(arr_t1c: np.ndarray, arr_t1c_raw: np.ndarray,
                   arr_t1n_raw: np.ndarray, n_comp: int = 4,
                   seed: int = 42) -> np.ndarray:
    """
    GMM 2D con features [T1c_limpio, mapa_dif].
    Identifica el cluster con T1c moderado-alto Y mapa_dif alto (= ET).
    Excluye vasos (mapa_dif > 0.40) y tejido oscuro (T1c bajo).
    """
    cerebro = _cerebro(arr_t1c)
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.5)

    t1c_vals = arr_t1c[cerebro]
    dif_vals = mapa[cerebro]
    X = np.stack([t1c_vals, dif_vals], axis=1).astype(np.float32)

    if X.shape[0] < n_comp:
        return np.zeros_like(arr_t1c, dtype=np.uint8)

    gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                          random_state=seed, max_iter=200)
    labels = gmm.fit_predict(X)

    mediana_t1c = float(np.median(t1c_vals))
    mejor_cluster = -1
    mejor_score = -np.inf
    for c in range(n_comp):
        mask_c = labels == c
        if not mask_c.any():
            continue
        t1c_mean = float(X[mask_c, 0].mean())
        dif_mean = float(X[mask_c, 1].mean())
        if dif_mean > 0.40 or t1c_mean < mediana_t1c * 0.8:
            continue
        score = t1c_mean * 0.4 + dif_mean * 0.6
        if score > mejor_score:
            mejor_score = score
            mejor_cluster = c

    if mejor_cluster == -1:
        mejor_cluster = int(np.argmax([
            X[labels == c, 1].mean() if (labels == c).any() else -np.inf
            for c in range(n_comp)
        ]))

    pred = np.zeros_like(arr_t1c, dtype=np.uint8)
    pred[cerebro] = (labels == mejor_cluster).astype(np.uint8)
    return _morfologia(pred, 1, 2, keep_largest=False)


# ================================================================== #
# Método 5: Rango doble en mapa T1c-T1n                             #
# ================================================================== #

def metodo_rango_doble(arr_t1c_raw: np.ndarray,
                        arr_t1n_raw: np.ndarray,
                        pct_bajo: float = 70.0,
                        umbral_alto: float = 0.55,
                        sigma: float = 0.5) -> np.ndarray:
    """
    Segmentacion ET por rango doble en el mapa T1c-T1n.

    Fundamento (observado en los datos):
      - ET:          mapa ~ 0.15 - 0.55  (realce tumoral moderado-alto)
      - Vasos:       mapa > 0.55         (realce extremo, excluir)
      - Ruido/sano:  mapa < 0.10         (sin realce, excluir)

    El rango doble [umbral_bajo, umbral_alto] aísla exactamente
    la señal tumoral excluyendo tanto el ruido bajo como los vasos.
    Esto es lo que un radiólogo hace visualmente en el mapa de sustracción.

    pct_bajo   : percentil del mapa positivo para umbral inferior (default 70)
    umbral_alto: techo fijo para excluir vasos (default 0.55)
    """
    cerebro = arr_t1c_raw > 0
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=sigma)

    # Umbral inferior: adaptativo por percentil
    vals_pos = mapa[cerebro & (mapa > 0)]
    if len(vals_pos) == 0:
        return np.zeros_like(arr_t1c_raw, dtype=np.uint8)
    umbral_bajo = float(np.percentile(vals_pos, pct_bajo))

    # Rango doble: excluye ruido bajo Y vasos altos
    pred = (mapa >= umbral_bajo) & (mapa <= umbral_alto) & cerebro

    # Limpiar componentes pequeños (< 30 vox = ruido puntual)
    struct = ndimage.generate_binary_structure(3, 1)
    labeled, n = ndimage.label(pred, structure=struct)
    if n > 0:
        sizes = ndimage.sum(pred, labeled, range(1, n + 1))
        pred = np.isin(labeled,
                       [i + 1 for i, s in enumerate(sizes) if s >= 30]
                       ).astype(np.uint8)

    return _morfologia(pred, 1, 1, keep_largest=False)


# ================================================================== #
# Método 6: FastMarching desde semilla (semi-automático)             #
# ================================================================== #

def metodo_fast_marching(arr_t1c_raw: np.ndarray,
                          arr_t1n_raw: np.ndarray,
                          semilla_zyx: Tuple[int, int, int],
                          tiempo_umbral: float = 35.0,
                          sigma: float = 0.8) -> np.ndarray:
    """
    Segmentacion ET por FastMarching con velocidad = mapa T1c-T1n.

    Principio (Ho et al., Prastawa et al.):
      El frente de onda FastMarching avanza con velocidad proporcional al
      mapa de diferencia T1c-T1n. En la region ET (mapa alto ~0.24) el
      frente avanza rapido -> tiempo de llegada bajo. En tejido sano
      (mapa bajo ~0.04) avanza lento -> tiempo alto. En bordes del ET
      el gradiente es alto -> la velocidad cae abruptamente -> el frente
      se detiene naturalmente.

      Umbralizar el mapa de tiempos de llegada aísla el ET sin depender
      de conectividad de intensidad (a diferencia del region growing).

    Ventaja sobre level sets:
      - Mas rapido y estable numericamente
      - Un solo parametro (tiempo_umbral) en vez de 4
      - Los vasos tienen velocidad alta pero son angostos -> el frente
        llega rapido pero con poco volumen acumulado

    tiempo_umbral: valor de corte en el mapa de tiempos (default 20.0)
      Valores tipicos segun diagnostico: 15-30 para ET en BraTS 2024
    """
    cerebro = _cerebro(arr_t1c_raw)

    # Mapa de diferencia con normalizacion conjunta
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=sigma)
    mapa_pos = np.clip(mapa, 0.001, None).astype(np.float32)  # evitar div/0

    # FastMarching con velocidad = mapa_pos
    speed_sitk = sitk.GetImageFromArray(mapa_pos)
    z0, y0, x0 = semilla_zyx
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint((int(x0), int(y0), int(z0), 0))  # orden ITK: x,y,z
    fm.SetStoppingValue(tiempo_umbral * 3.0)  # parar cuando no vale la pena
    tiempo_arr = sitk.GetArrayFromImage(fm.Execute(speed_sitk))

    # Umbralizar: tiempo bajo = region ET (avance rapido)
    pred = (tiempo_arr <= tiempo_umbral) & cerebro
    return _morfologia(pred.astype(np.uint8), 1, 1, keep_largest=True)


def buscar_umbral_fm(arr_t1c_raw: np.ndarray,
                      arr_t1n_raw: np.ndarray,
                      seg_gt_arr: np.ndarray,
                      semilla_zyx: Tuple[int, int, int],
                      umbral_grid: list = None) -> Tuple[float, float, dict]:
    """
    Busca el tiempo_umbral optimo para FastMarching sobre datos con GT.
    Solo usar en casos de entrenamiento/validacion.
    """
    cerebro = _cerebro(arr_t1c_raw)
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.8)
    mapa_pos = np.clip(mapa, 0.001, None).astype(np.float32)
    gt_et = (np.round(seg_gt_arr) == 3).astype(np.uint8)

    speed_sitk = sitk.GetImageFromArray(mapa_pos)
    z0, y0, x0 = semilla_zyx
    fm = sitk.FastMarchingImageFilter()
    fm.AddTrialPoint((int(x0), int(y0), int(z0), 0))
    fm.SetStoppingValue(1000.0)
    tiempo_arr = sitk.GetArrayFromImage(fm.Execute(speed_sitk))

    if umbral_grid is None:
        umbral_grid = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]

    resultados = {}
    for t in umbral_grid:
        pred = (tiempo_arr <= t) & cerebro
        d = dice(pred.astype(np.uint8), gt_et)
        resultados[t] = round(float(d), 4)

    mejor = max(resultados, key=resultados.get)
    return mejor, resultados[mejor], resultados


# ================================================================== #
# Método 7: Esfera + umbral (semi-automático, backup)                #
# ================================================================== #

def metodo_semilla(arr_t1c: np.ndarray,
                    arr_t1c_raw: np.ndarray,
                    arr_t1n_raw: np.ndarray,
                    semilla_zyx: Tuple[int, int, int],
                    radio_max: int = 25) -> np.ndarray:
    """
    Segmentacion ET basada en esfera + umbral de mapa_dif.

    Estrategia simple y robusta:
    1. Definir una esfera de radio_max alrededor de la semilla
    2. Dentro de esa esfera, tomar voxeles con mapa_dif > umbral_local
       (umbral = percentil 60 del mapa dentro de la esfera)
    3. Morfologia para limpiar

    Ventaja vs region growing: no se expande fuera de la esfera,
    no depende de conectividad — solo distancia + intensidad.
    El ET tipicamente cabe en una esfera de radio 25 voxeles.
    """
    cerebro = _cerebro(arr_t1c)
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    z0, y0, x0 = semilla_zyx
    sz = mapa.shape

    # Mascara de esfera
    zz = np.arange(sz[0])
    yy = np.arange(sz[1])
    xx = np.arange(sz[2])
    ZZ, YY, XX = np.meshgrid(zz, yy, xx, indexing='ij')
    dist = np.sqrt((ZZ - z0)**2 + (YY - y0)**2 + (XX - x0)**2)
    esfera = dist <= radio_max

    # Dentro de la esfera, umbral adaptativo del mapa
    vals_esfera = mapa[esfera & cerebro & (mapa > 0)]
    if len(vals_esfera) == 0:
        return np.zeros_like(arr_t1c, dtype=np.uint8)

    # Usar percentil 60 dentro de la esfera como umbral
    umbral = float(np.percentile(vals_esfera, 65))
    pred = (esfera & cerebro & (mapa >= umbral)).astype(np.uint8)

    return _morfologia(pred, 1, 2, keep_largest=True)


def semilla_automatica(arr_t1c: np.ndarray,
                        arr_t1c_raw: np.ndarray,
                        arr_t1n_raw: np.ndarray) -> Tuple[int, int, int]:
    """
    Semilla automatica para FastMarching sin usar el GT.

    Score = T1c_limpio * mapa_dif, excluyendo vasos (mapa > 0.55).
    Suavizado espacial para evitar voxeles aislados.
    El ET tiene T1c alto (~0.62) Y mapa alto (~0.24) -> score maximo.
    """
    from scipy.ndimage import uniform_filter
    cerebro = _cerebro(arr_t1c)
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma=0.5)
    mascara_valida = cerebro & (mapa > 0.05) & (mapa < 0.55)
    score = arr_t1c * mapa
    score_suav = uniform_filter(
        np.where(mascara_valida, score, 0).astype(np.float32), size=5)
    score_suav = np.where(mascara_valida, score_suav, -np.inf)
    idx = np.unravel_index(np.argmax(score_suav), score_suav.shape)
    return idx  # (z, y, x)


def correr_pipeline_et(
    t1c: sitk.Image,
    t1n: sitk.Image,
    seg_gt: sitk.Image,
    t1c_raw: sitk.Image = None,
    t1n_raw: sitk.Image = None,
    t2f: sitk.Image = None,
    semilla_zyx: Optional[Tuple[int, int, int]] = None,
    case_id: str = "",
    auto_pct: float = 90.0,
    sigma: float = 0.5,
    methods: Optional[list] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, object]:
    """
    Corre todos los métodos ET y devuelve máscaras + métricas.

    Parameters
    ----------
    t1c, t1n     : volúmenes limpios (Wiener + N4 + percentil)
    t1c_raw, t1n_raw : volúmenes crudos (sin N4) para sustracción y gmm_2d
    t2f          : FLAIR limpio (opcional, para ROI tumoral)
    semilla_zyx  : (z,y,x) colocada manualmente en el visor. Si se provee,
                   activa el método 'semilla' que puede dar Dice 0.5-0.8.
    """
    import pandas as pd

    pipeline_t0 = time.perf_counter()
    # WARNING: determinism fix. Keep both global seeds at pipeline entry so
    # GMM/Chan-Vese helper calls cannot inherit state from earlier cases.
    # Tests assert bit-identical masks across subprocess runs.
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    selected_methods = set(methods) if methods is not None else None

    def want(name: str) -> bool:
        return selected_methods is None or name in selected_methods

    method_times = {}

    def timed(name: str, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            method_times[name] = time.perf_counter() - t0

    arr_t1c = io_utils.a_numpy(t1c).astype(np.float32)
    arr_t1n = io_utils.a_numpy(t1n).astype(np.float32)

    # Raw arrays para sustracción (sin N4 preserva relación de escala)
    arr_t1c_raw = io_utils.a_numpy(t1c_raw).astype(np.float32) \
        if t1c_raw is not None else arr_t1c
    arr_t1n_raw = io_utils.a_numpy(t1n_raw).astype(np.float32) \
        if t1n_raw is not None else arr_t1n

    # GT resampleado al espacio de T1c limpio
    seg_r = sitk.Resample(sitk.Cast(seg_gt, sitk.sitkInt16), t1c,
                           sitk.Transform(), sitk.sitkNearestNeighbor,
                           0, sitk.sitkInt16)
    arr_seg = np.round(io_utils.a_numpy(seg_r)).astype(np.int16)
    gt_et = (arr_seg == config.LABEL_ET).astype(np.uint8)

    if verbose:
        print(f"\n  [{case_id}] GT-ET: {gt_et.sum()} vóxeles")

    # ── Métodos automáticos ────────────────────────────────────────
    pred_otsu = timed("otsu_T1c", lambda: metodo_otsu(arr_t1c)) if want("otsu_T1c") else None
    if verbose and pred_otsu is not None:
        print(f"  otsu         : pred={pred_otsu.sum():7d} vox")

    pred_gmm = timed("gmm_T1c", lambda: metodo_gmm(arr_t1c)) if want("gmm_T1c") else None
    if verbose and pred_gmm is not None:
        print(f"  gmm_T1c      : pred={pred_gmm.sum():7d} vox")

    if want("sustraccion"):
        pred_sust, mapa_dif, umbral = timed(
            "sustraccion",
            lambda: metodo_sustraccion(
                arr_t1c_raw, arr_t1n_raw, auto_pct=auto_pct, sigma=sigma),
        )
        if verbose and want("sustraccion"):
            print(f"  sustraccion  : pred={pred_sust.sum():7d} vox  umbral={umbral:.3f}")
    elif want("gmm_2d"):
        _, mapa_dif, umbral = metodo_sustraccion(
            arr_t1c_raw, arr_t1n_raw, auto_pct=auto_pct, sigma=sigma)
        pred_sust = None
    else:
        pred_sust = None
        mapa_dif = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma)

    pred_gmm2d = timed(
        "gmm_2d",
        lambda: metodo_gmm_2d(arr_t1c, arr_t1c_raw, arr_t1n_raw),
    ) if want("gmm_2d") else None
    if verbose and pred_gmm2d is not None:
        print(f"  gmm_2d       : pred={pred_gmm2d.sum():7d} vox")

    # ── Método 5: Rango doble ─────────────────────────────────────
    pred_rango = timed(
        "rango_doble",
        lambda: metodo_rango_doble(arr_t1c_raw, arr_t1n_raw),
    ) if want("rango_doble") else None
    if verbose and pred_rango is not None:
        print(f"  rango_doble  : pred={pred_rango.sum():7d} vox  [mapa en rango ET]")

    mascaras: Dict[str, np.ndarray] = {}
    for nombre, pred in [
        ("otsu_T1c", pred_otsu),
        ("gmm_T1c", pred_gmm),
        ("sustraccion", pred_sust),
        ("gmm_2d", pred_gmm2d),
        ("rango_doble", pred_rango),
    ]:
        if pred is not None:
            mascaras[nombre] = pred

    # ── Métodos de contornos deformables (spline / level set), automáticos ──
    from . import seg_spline_levelset as sls
    deformable_methods = [
        name for name in ("level_set", "variational_spline", "bspline", "spline")
        if want(name)
    ]
    guard_info = {}
    if deformable_methods:
        spl = sls.correr_spline_levelset(arr_t1c, arr_t1c_raw, arr_t1n_raw,
                                         sigma=sigma, methods=deformable_methods,
                                         verbose=verbose)
        guard_info = dict(sls.GUARD_INFO)
        method_times.update(getattr(sls, "METHOD_TIMINGS", {}))
        mascaras.update(spl)

    # ── FastMarching: semilla manual o automática ──────────────────
    # Siempre corre FastMarching - con semilla manual si se da, sino automática
    semilla_usada = semilla_zyx
    if want("fast_marching"):
        if semilla_usada is None:
            semilla_usada = semilla_automatica(arr_t1c, arr_t1c_raw, arr_t1n_raw)
            z0,y0,x0 = semilla_usada
            if verbose:
                print(f"  semilla auto : z={z0} y={y0} x={x0}  [score=T1c*mapa_dif]")

        pred_fm = timed(
            "fast_marching",
            lambda: metodo_fast_marching(
                arr_t1c_raw, arr_t1n_raw, semilla_usada,
                tiempo_umbral=config.FAST_MARCHING_TIME_THRESHOLD),
        )
        mascaras["fast_marching"] = pred_fm
        if verbose:
            z0,y0,x0 = semilla_usada
            modo = "manual" if semilla_zyx else "auto"
            print(f"  fast_marching: pred={pred_fm.sum():7d} vox  "
                  f"@ z={z0},y={y0},x={x0}  [{modo}]")

    # Esfera (backup, solo si hay semilla manual)
    if semilla_zyx is not None and want("semilla"):
        pred_seed = timed(
            "semilla",
            lambda: metodo_semilla(
                arr_t1c, arr_t1c_raw, arr_t1n_raw, semilla_zyx),
        )
        mascaras["semilla"] = pred_seed
        if verbose:
            print(f"  semilla      : pred={pred_seed.sum():7d} vox")

    # ── Métricas ──────────────────────────────────────────────────
    filas = []
    for nombre, pred in mascaras.items():
        guard = guard_info.get(nombre, {})
        d = dice(pred, gt_et)
        j = jaccard(pred, gt_et)
        filas.append({
            "case_id":    case_id,
            "metodo":     nombre,
            "dice_ET":    round(d, 4),
            "jaccard_ET": round(j, 4),
            "vol_GT":     int(gt_et.sum()),
            "vol_pred":   int(pred.sum()),
            "guard_branch": guard.get("branch", ""),
            "guard_reason": guard.get("reason", ""),
            "tiempo_s": round(float(method_times.get(nombre, 0.0)), 4),
        })
        if verbose:
            print(f"    → Dice={d:.3f}  Jaccard={j:.3f}  [{nombre}]")

    df = pd.DataFrame(filas)
    case_wall_s = time.perf_counter() - pipeline_t0
    method_total_s = float(sum(method_times.get(nombre, 0.0)
                               for nombre in mascaras
                               if not nombre.startswith("_")))
    df["case_wall_s"] = round(float(case_wall_s), 4)
    df["shared_preproc_s"] = round(max(0.0, case_wall_s - method_total_s), 4)

    # Guardar internos para visualización
    mascaras["_mapa_dif"]  = mapa_dif
    mascaras["_gt_et"]     = gt_et
    mascaras["_arr_t1c"]   = arr_t1c
    mascaras["_arr_t1c_raw"] = arr_t1c_raw
    mascaras["_arr_t1n_raw"] = arr_t1n_raw

    return mascaras, gt_et, df

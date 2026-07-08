"""
seg_spline_levelset.py
======================
Segmentación del Enhancing Tumor (ET, etiqueta 3) en BraTS 2024 GLI mediante
métodos de contornos deformables (spline / level set), TODOS automáticos.

Idea común
----------
El realce de gadolinio aparece en el mapa de diferencia  D = T1c − T1n
(normalización conjunta).  El ET es una masa de realce moderado-alto; los
vasos son finos y los plexos coroideos son pequeños focos.  Por eso:

  1.  `roi_et_auto`  obtiene una inicialización automática quedándose con la
      MASA de realce dominante (mayor  tamaño × realce medio), descartando
      vasos/plexos finos por tamaño.
  2.  Cuatro modelos deformables refinan ese contorno hacia el borde real del
      realce, cada uno con una regularización distinta:

         spline             — snake paramétrico de Kass (contorno = spline) por corte
         variational_spline — Chan-Vese morfológico (level set variacional, energía de región)
         bspline            — level set geodésico (SITK) + suavizado B-spline del contorno
         level_set          — level set geodésico de contornos activos (edge-based, SITK)

Todas devuelven una máscara binaria uint8 del ET.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import time
from scipy import ndimage
from scipy.interpolate import splprep, splev

import SimpleITK as sitk
from skimage.segmentation import morphsnakes as _morphsnakes
from skimage.segmentation import (
    morphological_chan_vese,
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
    active_contour,
)
from skimage.measure import find_contours
from skimage.draw import polygon as draw_polygon

from . import config
from .seg_et_pipeline import _mapa_diferencia, _morfologia, _cerebro, metodo_gmm


# ================================================================== #
# Utilidades de componentes                                          #
# ================================================================== #
_ST = ndimage.generate_binary_structure(3, 1)

# Si la semilla GMM supera este tamaño, la separación de intensidad falló
# (capta tejido sano brillante): el ET real rara vez excede ~150 k vox.  En ese
# caso se omite la costosa evolución del contorno y se devuelve la semilla
# (acota el tiempo de cómputo y evita que un level set llene medio cerebro).
MAX_SEED = 250_000
# El coste de los level sets lo fija el VOLUMEN DE LA CAJA (bbox), no el nº de
# vóxeles: una semilla dispersa da una caja enorme y vuelve la evolución lentísima.
# Si la caja supera este volumen, la semilla no es un tumor compacto -> usar semilla.
MAX_BBOX = 2_500_000
LAST_POST_INFO = {}
GUARD_INFO = {}
METHOD_TIMINGS = {}


def _reset_morphsnakes_curvop(start: str = "first") -> None:
    """Reset skimage's stateful Chan-Vese smoothing cycle.

    WARNING: scikit-image stores the morphological Chan-Vese curvature operator
    as a mutable module-level cycle. Without resetting it before each call, two
    identical runs in the same process can return different masks. This is the
    determinism fix pinned by ``tests/test_determinism.py``.
    """
    ops = [
        lambda u: _morphsnakes.sup_inf(_morphsnakes.inf_sup(u)),
        lambda u: _morphsnakes.inf_sup(_morphsnakes.sup_inf(u)),
    ]
    if start == "second":
        ops = [ops[1], ops[0]]
    _morphsnakes._curvop = _morphsnakes._fcycle(ops)  # noqa: SLF001


def _semilla_degenerada(roi: np.ndarray) -> bool:
    if int(roi.sum()) > MAX_SEED:
        return True
    if not roi.any():
        return False
    coords = np.argwhere(roi)
    ext = coords.max(0) - coords.min(0) + 1
    return int(np.prod(ext)) > MAX_BBOX


def _keep_components_min(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=_ST)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    return np.isin(lab, keep).astype(np.uint8)


def _bbox(mask: np.ndarray, margin: int, shape) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


# ================================================================== #
# Inicialización automática: la masa de realce dominante             #
# ================================================================== #
def _seed_diferencia(mapa: np.ndarray, cerebro: np.ndarray,
                     pct: float = 90.0) -> np.ndarray:
    """Masa de realce dominante del mapa T1c−T1n (componente que maximiza
    tamaño × realce medio tras umbralizar el mapa positivo)."""
    vals = mapa[cerebro & (mapa > 0)]
    if vals.size == 0:
        return np.zeros_like(mapa, dtype=np.uint8)
    cand = ndimage.binary_closing(
        (mapa >= float(np.percentile(vals, pct))) & cerebro,
        structure=_ST, iterations=1)
    lab, n = ndimage.label(cand, structure=_ST)
    mejor, mejor_score = 0, -np.inf
    for i in range(1, n + 1):
        comp = lab == i
        size = int(comp.sum())
        if size < 80:
            continue
        score = size * float(mapa[comp].mean())
        if score > mejor_score:
            mejor_score, mejor = score, i
    if mejor == 0:
        return np.zeros_like(mapa, dtype=np.uint8)
    roi = ndimage.binary_fill_holes(lab == mejor)
    return ndimage.binary_dilation(roi, structure=_ST,
                                   iterations=1).astype(np.uint8)


def roi_et_auto(arr_t1c: np.ndarray,
                arr_t1c_raw: np.ndarray,
                arr_t1n_raw: np.ndarray,
                sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (roi_init, mapa_dif) con una **semilla híbrida**.

    Ninguna señal aislada es robusta para el ET:
      * GMM sobre T1c localiza la masa realzante aunque el mapa T1c−T1n sea
        débil/mal registrado, pero degenera (capta tejido sano brillante) en
        casos de bajo contraste.
      * El blob dominante del mapa T1c−T1n aísla bien el realce, pero falla
        cuando el mapa es débil.

    Estrategia: calcular ambas semillas y elegir la de tamaño plausible
    (80…MAX_SEED vox) con mayor **masa realzante** (tamaño × realce medio).
    Así cada caso usa la señal que funciona, y los modelos deformables refinan
    esa semilla (con reversión de seguridad en `_post`).
    """
    cerebro = arr_t1c_raw > 0
    mapa = _mapa_diferencia(arr_t1c_raw, arr_t1n_raw, sigma)

    s_gmm = metodo_gmm(arr_t1c, n_comp=3)
    if s_gmm.any():
        s_gmm = ndimage.binary_fill_holes(s_gmm).astype(np.uint8)
    sz = int(s_gmm.sum())

    # GMM-primario: si su semilla es de tamaño plausible, es la más fiable
    # (localiza la masa realzante aun con mapa de diferencia débil).  Solo
    # cuando GMM degenera (vacía o > MAX_SEED, capta tejido sano brillante) se
    # recurre al blob dominante del mapa T1c−T1n.
    if 80 < sz <= MAX_SEED:
        seed = s_gmm
    else:
        s_dif = _seed_diferencia(mapa, cerebro)
        seed = s_dif if s_dif.any() else s_gmm
    return seed.astype(np.uint8), mapa


def _restringir_realce(pred: np.ndarray, mapa: np.ndarray,
                       cerebro: np.ndarray, pct: float = 80.0) -> np.ndarray:
    """Intersecta la predicción con voxeles de realce real (evita fugas al
    tejido no realzante que el contorno pudiera invadir)."""
    vals = mapa[cerebro & (mapa > 0)]
    if vals.size == 0:
        return pred
    umbral = float(np.percentile(vals, pct))
    return ((pred > 0) & (mapa >= umbral)).astype(np.uint8)


def _orientar_chanvese(ls: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Chan-Vese puede etiquetar como 'interior' la región oscura. Nos quedamos
    con la fase cuya intensidad media es mayor (el realce del ET es brillante)."""
    if not ls.any() or ls.all():
        return ls
    if img[ls == 1].mean() < img[ls == 0].mean():
        ls = 1 - ls
    return ls.astype(np.uint8)


def _largest_component_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    lab, n = ndimage.label(mask > 0, structure=_ST)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(mask > 0, lab, range(1, n + 1))
    return float(np.max(sizes) / float(mask.sum()))


def _evidence_accepts_evolved(pred: np.ndarray, init: np.ndarray,
                              mapa: np.ndarray) -> Tuple[bool, dict]:
    """Return whether the Stage 3A evidence guard accepts seed-divergent growth.

    WARNING: this is not a generic cleanup guard. It is a frozen study rule that
    preserves the evolved 02116 mask instead of forcing the ROI fallback. Any
    change must be treated as a numerical-method change and baseline refreeze.
    """
    a_init = float(init.sum()) if init is not None else 0.0
    a_pred = float(pred.sum())
    pred_vals = mapa[pred > 0]
    init_vals = mapa[init > 0] if init is not None else np.array([], dtype=mapa.dtype)
    pred_enh = float(pred_vals.mean()) if pred_vals.size else 0.0
    init_enh = float(init_vals.mean()) if init_vals.size else 0.0
    lcc_fraction = _largest_component_fraction(pred)
    volume_multiple = a_pred / max(a_init, 1.0)
    enhancement_ratio = pred_enh / max(init_enh, 1e-6)
    accepted = (
        config.ENABLE_EVIDENCE_GUARD
        and lcc_fraction >= config.GUARD_MIN_LCC_FRACTION
        and enhancement_ratio >= config.GUARD_MIN_ENHANCEMENT_RATIO
        and volume_multiple <= config.GUARD_MAX_VOLUME_MULTIPLE
    )
    return accepted, {
        "evidence_lcc_fraction": lcc_fraction,
        "evidence_pred_enhancement": pred_enh,
        "evidence_init_enhancement": init_enh,
        "evidence_enhancement_ratio": enhancement_ratio,
        "evidence_volume_multiple": volume_multiple,
        "evidence_accept": bool(accepted),
    }


def _chanvese_iterate_score(pred: np.ndarray, init: np.ndarray, mapa: np.ndarray,
                            prev_voxels: int = None) -> Tuple[bool, float, dict]:
    a_init = float(init.sum()) if init is not None else 0.0
    a_pred = float(pred.sum())
    pred_vals = mapa[pred > 0]
    init_vals = mapa[init > 0] if init is not None else np.array([], dtype=mapa.dtype)
    pred_enh = float(pred_vals.mean()) if pred_vals.size else 0.0
    init_enh = float(init_vals.mean()) if init_vals.size else 0.0
    enhancement_ratio = pred_enh / max(init_enh, 1e-6)
    volume_multiple = a_pred / max(a_init, 1.0)
    lcc_fraction = _largest_component_fraction(pred)
    volume_stability = 0.0
    if prev_voxels is not None and prev_voxels > 0:
        volume_stability = max(0.0, 1.0 - abs(a_pred - prev_voxels) / float(prev_voxels))

    acceptable = bool(a_pred > 0 and init is not None and init.any())
    if acceptable:
        inter = float(np.logical_and(pred > 0, init > 0).sum())
        if a_pred < 0.40 * a_init:
            acceptable = False
        elif a_pred > config.GUARD_MAX_VOLUME_MULTIPLE * a_init:
            acceptable = False
        elif inter < 0.40 * a_pred:
            acceptable = (
                lcc_fraction >= config.GUARD_MIN_LCC_FRACTION
                and enhancement_ratio >= config.GUARD_MIN_ENHANCEMENT_RATIO
                and volume_multiple <= config.GUARD_MAX_VOLUME_MULTIPLE
            )

    score = (
        config.BEST_ITERATE_W_LCC * lcc_fraction
        + config.BEST_ITERATE_W_ENHANCEMENT * min(enhancement_ratio, 1.5)
        + config.BEST_ITERATE_W_VOLUME_STABILITY * volume_stability
    )
    return acceptable, float(score), {
        "best_lcc_fraction": lcc_fraction,
        "best_pred_enhancement": pred_enh,
        "best_init_enhancement": init_enh,
        "best_enhancement_ratio": enhancement_ratio,
        "best_volume_multiple": volume_multiple,
        "best_volume_stability": volume_stability,
    }


def _select_best_chanvese_iterate(img: np.ndarray, init: np.ndarray, mapa: np.ndarray,
                                  sl: Tuple[slice, slice, slice], iters: int,
                                  smoothing: int, lambda1: float = 1.0,
                                  lambda2: float = 1.0) -> Tuple[np.ndarray, dict]:
    snapshots = {}
    callback_count = {"i": 0}

    def callback(u):
        snapshots[callback_count["i"]] = np.array(u, copy=True)
        callback_count["i"] += 1

    _reset_morphsnakes_curvop("first")
    final = morphological_chan_vese(
        img, num_iter=iters, init_level_set=init,
        smoothing=smoothing, lambda1=lambda1, lambda2=lambda2,
        iter_callback=callback).astype(np.uint8)

    best_ls = final
    best_score = -np.inf
    best_info = {"best_iter": iters, "best_score": float("nan"), "best_acceptable": False}
    prev_voxels = None
    drops = 0
    for iteration in sorted(snapshots):
        ls = _orientar_chanvese(snapshots[iteration].astype(np.uint8), img)
        pred = np.zeros_like(mapa, dtype=np.uint8)
        pred[sl] = ls
        acceptable, score, info = _chanvese_iterate_score(pred, init=np.pad(
            init, [(sl[i].start, mapa.shape[i] - sl[i].stop) for i in range(3)],
            mode="constant"), mapa=mapa, prev_voxels=prev_voxels)
        voxels = int(pred.sum())
        prev_voxels = voxels
        if iteration > 0 and acceptable and score > best_score:
            best_ls = ls
            best_score = score
            best_info = {
                "best_iter": int(iteration),
                "best_score": float(score),
                "best_acceptable": True,
                "best_voxels": voxels,
                **info,
            }
            drops = 0
        elif iteration > 0 and best_score > -np.inf and score < best_score:
            drops += 1
            if drops >= config.BEST_ITERATE_PATIENCE:
                break
        else:
            drops = 0

    if not best_info["best_acceptable"]:
        best_ls = init
        best_info = {
            "best_iter": 0,
            "best_score": float("nan"),
            "best_acceptable": False,
            "best_voxels": int(init.sum()),
            "best_fallback": "roi_no_acceptable_iterate",
        }
    return best_ls.astype(np.uint8), best_info


def _post(pred: np.ndarray, mapa: np.ndarray, cerebro: np.ndarray,
          init: np.ndarray = None, pct_realce: float = 80.0) -> np.ndarray:
    """Post-proceso común con salvaguardas, partiendo de la semilla GMM `init`.

    1.  Restricción a realce SOLO si conserva ≥50 % de la predicción (cuando el
        mapa de diferencia es informativo elimina fugas; cuando es débil no se
        aplica, para no vaciar una buena semilla).
    2.  Si el contorno colapsó (<30 % de la semilla) o se fugó (>3×), se revierte
        a la semilla GMM cruda — así un modelo deformable degenerado nunca
        puntúa por debajo de la base GMM.
    """
    global LAST_POST_INFO
    raw = (pred > 0).astype(np.uint8)
    restr = _restringir_realce(raw, mapa, cerebro, pct_realce)
    used_restriction = bool(raw.sum() > 0 and restr.sum() >= 0.5 * raw.sum())
    pred = restr if used_restriction else raw
    fallback_reason = ""
    evidence_accepted_seed_divergence = False
    evidence_info = {
        "evidence_lcc_fraction": 0.0,
        "evidence_pred_enhancement": 0.0,
        "evidence_init_enhancement": 0.0,
        "evidence_enhancement_ratio": 0.0,
        "evidence_volume_multiple": 0.0,
        "evidence_accept": False,
    }

    if init is not None and init.any():
        a_init = float(init.sum())
        a_pred = float(pred.sum())
        inter = float(np.logical_and(pred > 0, init > 0).sum())
        # Revertir si: colapsó, se fugó en volumen, o se fue a OTRA región
        # (área parecida pero poco solape con la semilla GMM).
        if a_pred == 0:
            fallback_reason = "empty_prediction"
        elif a_pred < 0.40 * a_init:
            fallback_reason = "collapsed_small"
        elif a_pred > config.GUARD_MAX_VOLUME_MULTIPLE * a_init:
            fallback_reason = "leaked_large"
        elif inter < 0.40 * a_pred:
            evidence_accept, evidence_info = _evidence_accepts_evolved(pred, init, mapa)
            if evidence_accept:
                evidence_accepted_seed_divergence = True
            else:
                fallback_reason = "low_seed_overlap"
        if fallback_reason:
            pred = (init > 0).astype(np.uint8)

    if pred.any():
        pred = ndimage.binary_fill_holes(pred).astype(np.uint8)
        pred = _morfologia(pred, erosion=0, dilatacion=0, keep_largest=True)
    equals_init = bool(init is not None and init.any()
                       and pred.shape == init.shape
                       and np.array_equal(pred > 0, init > 0))
    LAST_POST_INFO = {
        "branch": "collapse-detected" if fallback_reason else
                  "ROI-fallback" if equals_init else "evolved",
        "reason": fallback_reason if fallback_reason else
                  "final_equals_init_no_collapse" if equals_init else
                  "accepted_evidence_seed_divergence" if evidence_accepted_seed_divergence else
                  "accepted",
        "used_restriction": used_restriction,
        "raw_voxels": int(raw.sum()),
        "restricted_voxels": int(restr.sum()),
        "final_voxels": int(pred.sum()),
        "init_voxels": int(init.sum()) if init is not None else 0,
        **evidence_info,
    }
    return pred.astype(np.uint8)


# ================================================================== #
# Método A: level_set — Geodesic Active Contour (SimpleITK)          #
# ================================================================== #
def metodo_level_set(arr_t1c: np.ndarray,
                     arr_t1c_raw: np.ndarray,
                     arr_t1n_raw: np.ndarray,
                     sigma: float = 0.5,
                     prop: float = 0.8,
                     curv: float = 3.0,
                     adv: float = 1.5,
                     iters: int = None,
                     roi: np.ndarray = None,
                     mapa: np.ndarray = None) -> np.ndarray:
    """
    Level set geodésico (contornos activos basados en bordes).
    El frente parte de la ROI automática y evoluciona hacia los bordes del
    realce (imagen de velocidad sigmoide del gradiente del mapa de diferencia).
    """
    cerebro = arr_t1c_raw > 0
    if roi is None or mapa is None:
        roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma)
    if not roi.any():
        return roi
    if _semilla_degenerada(roi):
        return _post((roi > 0).astype(np.uint8), mapa, cerebro, init=roi)
    iters = config.LEVEL_SET_ITERS if iters is None else iters

    sl = _bbox(roi, margin=12, shape=mapa.shape)
    mapa_c = mapa[sl].astype(np.float32)
    roi_c = roi[sl].astype(np.uint8)

    # Imagen de bordes: gradiente del mapa -> sigmoide (velocidad ~0 en bordes).
    feat = sitk.GetImageFromArray(mapa_c)
    grad = sitk.GradientMagnitudeRecursiveGaussian(feat, sigma=1.0)
    speed = sitk.Sigmoid(grad, alpha=-0.05, beta=0.1,
                         outputMaximum=1.0, outputMinimum=0.0)

    # Inicial: distancia con signo (negativa dentro de la ROI).
    init = sitk.SignedMaurerDistanceMap(
        sitk.GetImageFromArray(roi_c), insideIsPositive=False,
        squaredDistance=False, useImageSpacing=False)

    gac = sitk.GeodesicActiveContourLevelSetImageFilter()
    gac.SetPropagationScaling(prop)
    gac.SetCurvatureScaling(curv)
    gac.SetAdvectionScaling(adv)
    gac.SetMaximumRMSError(0.01)
    gac.SetNumberOfIterations(iters)
    out = gac.Execute(sitk.Cast(init, sitk.sitkFloat32),
                      sitk.Cast(speed, sitk.sitkFloat32))
    phi = sitk.GetArrayFromImage(out)

    pred = np.zeros_like(mapa, dtype=np.uint8)
    pred[sl] = (phi < 0).astype(np.uint8)
    return _post(pred, mapa, cerebro, init=roi, pct_realce=80.0)


# ================================================================== #
# Método B: variational_spline — Chan-Vese morfológico (level set)   #
# ================================================================== #
def metodo_variational_spline(arr_t1c: np.ndarray,
                              arr_t1c_raw: np.ndarray,
                              arr_t1n_raw: np.ndarray,
                              sigma: float = 0.5,
                              iters: int = None,
                              smoothing: int = None,
                              roi: np.ndarray = None,
                              mapa: np.ndarray = None) -> np.ndarray:
    """
    Level set variacional (Chan-Vese morfológico): minimiza la energía de
    región de Mumford-Shah sobre el mapa de diferencia, partiendo de la ROI.
    `smoothing` aplica regularización de curvatura (suavidad tipo spline).
    """
    cerebro = arr_t1c_raw > 0
    if roi is None or mapa is None:
        roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma)
    if not roi.any():
        return roi
    if _semilla_degenerada(roi):
        return _post((roi > 0).astype(np.uint8), mapa, cerebro, init=roi)
    iters = config.VARIATIONAL_SPLINE_ITERS if iters is None else iters
    smoothing = config.VARIATIONAL_SPLINE_SMOOTHING if smoothing is None else smoothing

    sl = _bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype(np.uint8)

    best_info = {}
    if config.ENABLE_BEST_ITERATE:
        ls, best_info = _select_best_chanvese_iterate(
            img, init, mapa, sl, iters=iters, smoothing=smoothing,
            lambda1=1.0, lambda2=1.0)
    else:
        _reset_morphsnakes_curvop("first")
        ls = morphological_chan_vese(
            img, num_iter=iters, init_level_set=init,
            smoothing=smoothing, lambda1=1.0, lambda2=1.0).astype(np.uint8)
        ls = _orientar_chanvese(ls, img)

    pred = np.zeros_like(mapa, dtype=np.uint8)
    pred[sl] = ls
    out = _post(pred, mapa, cerebro, init=roi, pct_realce=80.0)
    if best_info:
        LAST_POST_INFO.update(best_info)
    return out


# ================================================================== #
# Método C: bspline — GAC + suavizado B-spline del contorno          #
# ================================================================== #
def _suavizar_bspline_slice(mask2d: np.ndarray, smooth: float = 2.0,
                            min_pts: int = 12) -> np.ndarray:
    """Re-dibuja una máscara 2D suavizando su contorno externo con un
    B-spline periódico (scipy.splprep)."""
    if mask2d.sum() < min_pts:
        return mask2d
    conts = find_contours(mask2d.astype(float), 0.5)
    if not conts:
        return mask2d
    cont = max(conts, key=len)              # contorno externo principal
    if len(cont) < min_pts:
        return mask2d
    y, x = cont[:, 0], cont[:, 1]
    try:
        # B-spline cúbico periódico; s controla la suavidad.
        tck, _ = splprep([x, y], s=len(x) * smooth, per=True, k=3)
        u = np.linspace(0, 1, max(len(x), 60))
        xs, ys = splev(u, tck)
    except Exception:
        return mask2d
    out = np.zeros_like(mask2d, dtype=np.uint8)
    rr, cc = draw_polygon(ys, xs, shape=mask2d.shape)
    out[rr, cc] = 1
    return out


def metodo_bspline(arr_t1c: np.ndarray,
                   arr_t1c_raw: np.ndarray,
                   arr_t1n_raw: np.ndarray,
                   sigma: float = 0.5,
                   roi: np.ndarray = None,
                   mapa: np.ndarray = None) -> np.ndarray:
    """
    Superficie B-spline: parte del level set variacional de región (Chan-Vese
    morfológico) y regulariza su frontera corte a corte con un B-spline cúbico
    periódico (scipy.splprep).  El B-spline impone un borde liso (continuidad
    C2, superficie spline) sobre el resultado del contorno activo, suprimiendo
    el dentado del marching-cubes implícito.
    """
    cerebro = arr_t1c_raw > 0
    if roi is None or mapa is None:
        roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma)
    if not roi.any():
        return roi
    if _semilla_degenerada(roi):
        return _post((roi > 0).astype(np.uint8), mapa, cerebro, init=roi)
    iters = config.BSPLINE_CHANVESE_ITERS
    smoothing = config.BSPLINE_CHANVESE_SMOOTHING

    sl = _bbox(roi, margin=12, shape=mapa.shape)
    img = mapa[sl].astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    init = roi[sl].astype(np.uint8)

    # Base: level set variacional de región (robusto, no se infla como MorphGAC).
    _reset_morphsnakes_curvop("second")
    ls = morphological_chan_vese(
        img, num_iter=iters, init_level_set=init,
        smoothing=smoothing, lambda1=1.0, lambda2=1.0).astype(np.uint8)
    ls = _orientar_chanvese(ls, img)

    # Regularización B-spline del contorno por corte axial.
    for z in range(ls.shape[0]):
        if ls[z].any():
            ls[z] = _suavizar_bspline_slice(ls[z], smooth=3.0)

    pred = np.zeros_like(mapa, dtype=np.uint8)
    pred[sl] = ls
    return _post(pred, mapa, cerebro, init=roi, pct_realce=78.0)


# ================================================================== #
# Método D: spline — snake paramétrico de Kass por corte             #
# ================================================================== #
def metodo_spline(arr_t1c: np.ndarray,
                  arr_t1c_raw: np.ndarray,
                  arr_t1n_raw: np.ndarray,
                  sigma: float = 0.5,
                  roi: np.ndarray = None,
                  mapa: np.ndarray = None) -> np.ndarray:
    """
    Contorno activo paramétrico (snake de Kass): en cada corte axial con
    realce, se inicia un contorno spline alrededor de la ROI y se ajusta al
    borde del realce con `skimage.active_contour`.  La curva resultante es,
    por definición, un spline cerrado.
    """
    cerebro = arr_t1c_raw > 0
    if roi is None or mapa is None:
        roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma)
    if not roi.any():
        return roi
    if _semilla_degenerada(roi):
        return _post((roi > 0).astype(np.uint8), mapa, cerebro, init=roi)

    sl = _bbox(roi, margin=8, shape=mapa.shape)
    roi_c = roi[sl]
    pred_c = np.zeros_like(roi_c, dtype=np.uint8)
    # El snake "ve" el mapa de realce suavizado (atracción a brillo).
    img_c = ndimage.gaussian_filter(mapa[sl].astype(np.float32), 1.5)

    # Init un poco DILATADA: el snake (sin fuerza globo) tiende a contraerse,
    # así aterriza sobre el borde real del realce en vez de colapsar.
    roi_dil = ndimage.binary_dilation(roi_c, structure=_ST, iterations=2)

    # Por velocidad: si el tumor abarca muchos cortes, procesar con paso 2 y
    # rellenar los cortes saltados con una dilatación a lo largo de z.
    zs_all = np.where(roi_c.any(axis=(1, 2)))[0]
    stride = 2 if len(zs_all) > 40 else 1
    for z in zs_all[::stride]:
        m = roi_c[z]
        area = int(m.sum())
        if area < 30:
            continue
        md = roi_dil[z]
        conts = find_contours(md.astype(float), 0.5)
        if not conts:
            pred_c[z] = m            # fallback: la ROI del corte
            continue
        init = max(conts, key=len)
        if len(init) > 40:
            idx = np.linspace(0, len(init) - 1, 40).astype(int)
            init = init[idx]
        if len(init) < 10:
            pred_c[z] = m
            continue
        try:
            snake = active_contour(
                img_c[z], init, alpha=0.05, beta=2.0,
                w_line=2.0, w_edge=1.0, gamma=0.02,
                max_num_iter=25, boundary_condition="periodic")
            rr, cc = draw_polygon(snake[:, 0], snake[:, 1], shape=m.shape)
            cur = np.zeros_like(m, dtype=np.uint8)
            cur[rr, cc] = 1
            # Salvaguarda anti-colapso: si el snake se contrajo a <40% del
            # área de la ROI, usar la ROI del corte (el spline degeneró).
            if cur.sum() < 0.4 * area:
                cur = m.astype(np.uint8)
            pred_c[z] = cur
        except Exception:
            pred_c[z] = m

    if stride > 1:                       # rellenar cortes saltados
        zstruct = np.zeros((3, 3, 3), bool); zstruct[:, 1, 1] = True
        pred_c = (ndimage.binary_dilation(pred_c, structure=zstruct,
                                          iterations=1) & roi_dil).astype(np.uint8)

    pred = np.zeros_like(mapa, dtype=np.uint8)
    pred[sl] = pred_c
    return _post(pred, mapa, cerebro, init=roi, pct_realce=80.0)


# ================================================================== #
# Driver: corre los 4 métodos y devuelve dict de máscaras            #
# ================================================================== #
def correr_spline_levelset(arr_t1c: np.ndarray,
                           arr_t1c_raw: np.ndarray,
                           arr_t1n_raw: np.ndarray,
                           sigma: float = 0.5,
                           methods = None,
                           verbose: bool = True) -> dict:
    global LAST_POST_INFO, GUARD_INFO, METHOD_TIMINGS
    out = {}
    GUARD_INFO = {}
    METHOD_TIMINGS = {}
    # Semilla híbrida (incluye un ajuste GMM costoso): se calcula UNA vez y se
    # comparte entre los 4 métodos, en vez de recomputarla por método.
    roi, mapa = roi_et_auto(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma)
    if verbose:
        print(f"  semilla auto : {int(roi.sum())} vox"
              f"{'  [degenerada→semilla]' if _semilla_degenerada(roi) else ''}")
    metodos = [
        ("level_set",          metodo_level_set),
        ("variational_spline", metodo_variational_spline),
        ("bspline",            metodo_bspline),
        ("spline",             metodo_spline),
    ]
    selected = set(methods) if methods is not None else {nombre for nombre, _ in metodos}
    for nombre, fn in metodos:
        if nombre not in selected:
            continue
        LAST_POST_INFO = {}
        t_method = time.perf_counter()
        try:
            pred = fn(arr_t1c, arr_t1c_raw, arr_t1n_raw, sigma=sigma,
                      roi=roi, mapa=mapa)
            GUARD_INFO[nombre] = dict(LAST_POST_INFO) if LAST_POST_INFO else {
                "branch": "ROI-fallback" if np.array_equal(pred > 0, roi > 0) else "evolved",
                "reason": "no_post_info",
            }
        except Exception as e:
            if verbose:
                print(f"  [!] {nombre} falló: {e}")
            pred = np.zeros_like(arr_t1c, dtype=np.uint8)
            GUARD_INFO[nombre] = {
                "branch": "collapse-detected",
                "reason": f"exception:{type(e).__name__}",
            }
        METHOD_TIMINGS[nombre] = time.perf_counter() - t_method
        out[nombre] = pred
        if verbose:
            print(f"  {nombre:18s}: pred={int(pred.sum()):7d} vox")
    return out

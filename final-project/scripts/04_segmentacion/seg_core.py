"""
seg_core.py
===========
Núcleo de segmentación clásica del Tumor Realzante (ET) sobre T1c LIMPIO. Cosecha las
"joyas" del paquete de Santiago, adaptadas a nuestro objetivo (ET = label 3, T1c):

  * segmentar_otsu        <- seg_otsu.py        (Otsu multinivel n=3, clase alta)
  * connected_threshold / confidence_connected  <- seg_region_growing.py
  * segmentar_watershed   <- seg_watershed.py   (watershed marcador-controlado)
  * segmentar_clustering  <- seg_kmeans_gmm.py  (K-means / GMM, MULTIMODAL)

Todas operan sobre los volúmenes LIMPIOS ([0,1]) y devuelven máscara binaria uint8
(z,y,x) salvo las de SimpleITK, que reciben/parten de imágenes SITK.

Convención de semilla: `centroide_et_index` devuelve el índice IJK de SimpleITK
(orden x,y,z), que es el que esperan ConnectedThreshold/ConfidenceConnected.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.filters import sobel, threshold_multiotsu
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import constantes as C   # noqa: E402
from comun import io_zip            # noqa: E402


# --------------------------------------------------------------------------- #
# Semilla (centroide del ET del GT) e índices
# --------------------------------------------------------------------------- #
def centroide_et_index(seg: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Índice IJK de SimpleITK (x,y,z) de una semilla DENTRO del ET (label 3). None si no hay ET.

    Se parte del centroide del ET (`center_of_mass`, orden z,y,x). En ET post-tratamiento
    el realce suele tener forma de anillo alrededor de la cavidad de resección, por lo que
    el centroide geométrico puede caer FUERA del ET (en tejido oscuro) y arruinar el region
    growing. Por eso, si el centroide no cae en ET, se "salta" al vóxel de ET más cercano:
    así la semilla queda siempre dentro del realce. `center_of_mass` da (z,y,x) -> se invierte
    a (x,y,z) para SimpleITK.
    """
    et = seg == C.LABEL_ET
    if not et.any():
        return None
    cz, cy, cx = [int(round(v)) for v in ndi.center_of_mass(et)]
    if not et[cz, cy, cx]:
        # snap al vóxel de ET más cercano al centroide
        et_idx = np.argwhere(et)                       # (N, 3) en orden (z, y, x)
        d2 = ((et_idx - np.array([cz, cy, cx])) ** 2).sum(axis=1)
        cz, cy, cx = (int(v) for v in et_idx[int(d2.argmin())])
    return (cx, cy, cz)


def semilla_et_brillante(volumen: np.ndarray, seg: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Semilla de region growing para ET: el vóxel de ET (label 3) MÁS BRILLANTE en T1c.

    Para ET (realce) la semilla debe caer sobre tejido claramente realzado. El centroide
    del ET no sirve: en ET post-tratamiento (anillo alrededor de la cavidad) cae en tejido
    oscuro y el crecimiento se va por tejido equivocado. El vóxel de ET más brillante está
    sobre el realce y permite crecer la región enhancing. Devuelve IJK SITK (x,y,z) o None.
    """
    et = seg == C.LABEL_ET
    if not et.any():
        return None
    ev = np.where(et, volumen, -np.inf)
    z, y, x = np.unravel_index(int(np.argmax(ev)), ev.shape)
    return (int(x), int(y), int(z))


def semilla_fallback(volumen: np.ndarray) -> Tuple[int, int, int]:
    """
    Semilla automática cuando no hay ET en el GT: vóxel más brillante dentro del
    cerebro (suavizado para robustez). Devuelve índice IJK de SimpleITK (x,y,z).
    Sirve para verificar que los métodos sembrados CORREN aunque no haya ET.
    """
    v = np.where(volumen > 0, volumen, 0)
    v = ndi.gaussian_filter(v, sigma=1.0)
    z, y, x = np.unravel_index(int(np.argmax(v)), v.shape)
    return (int(x), int(y), int(z))


def coef_bimodalidad(valores: np.ndarray) -> float:
    """
    Coeficiente de bimodalidad de Sarle: BC = (skew^2 + 1) / (kurtosis_excess + 3).
    BC > 5/9 sugiere histograma bimodal; por debajo, unimodal (Otsu poco fiable).
    """
    v = np.asarray(valores, dtype=np.float64)
    if v.size < 4:
        return 0.0
    g = skew(v)
    k = kurtosis(v, fisher=True)   # exceso de curtosis
    return float((g ** 2 + 1.0) / (k + 3.0)) if (k + 3.0) > 1e-8 else 0.0


# --------------------------------------------------------------------------- #
# Método 1 — Otsu multinivel n=3, clase alta (de seg_otsu.py de Santiago)
# --------------------------------------------------------------------------- #
def segmentar_otsu(volumen: np.ndarray, clases: int = 3, tomar_clase: str = "alta",
                   nombre: str = "") -> Tuple[np.ndarray, dict]:
    """
    Otsu multinivel sobre T1c limpio. Toma la clase más brillante (realce). Recalcula
    el umbral sobre el volumen recibido (robusto a normalización). Avisa si el
    histograma es unimodal (límite conocido de Otsu, lo pide el enunciado).

    Devuelve (máscara uint8, info{bimodalidad, es_bimodal, umbral_alto}).
    """
    v = np.asarray(volumen, dtype=np.float32)
    cerebro = v != 0
    vals = v[cerebro]
    info = {"bimodalidad": np.nan, "es_bimodal": None, "umbral_alto": np.nan}
    if vals.size < clases or np.unique(vals).size < clases:
        print(f"  [AVISO] {nombre}: pocos niveles para Otsu n={clases}; máscara vacía.")
        return np.zeros_like(v, dtype=np.uint8), info

    bc = coef_bimodalidad(vals)
    info["bimodalidad"] = round(bc, 4)
    info["es_bimodal"] = bool(bc >= C.BIMODALIDAD_MIN)
    if not info["es_bimodal"]:
        print(f"  [AVISO] {nombre}: histograma UNIMODAL (BC={bc:.3f} < {C.BIMODALIDAD_MIN:.3f}); "
              f"Otsu poco fiable para ET en T1c.")

    umbrales = threshold_multiotsu(vals, classes=clases)
    info["umbral_alto"] = round(float(umbrales[-1]), 5)
    etiquetas = np.digitize(v, bins=umbrales)
    if tomar_clase == "alta":
        mask = etiquetas == (clases - 1)
    elif tomar_clase == "dos_altas":
        mask = etiquetas >= (clases - 2)
    else:
        raise ValueError("tomar_clase debe ser 'alta' o 'dos_altas'")
    mask &= cerebro
    return mask.astype(np.uint8), info


# --------------------------------------------------------------------------- #
# Método 2 — Crecimiento de regiones (de seg_region_growing.py de Santiago)
# --------------------------------------------------------------------------- #
def _estadisticos_semilla(img: sitk.Image, seed: Tuple[int, int, int],
                          radio: int = 2) -> Tuple[float, float]:
    """Media y std en una ventana cúbica (2*radio+1)^3 alrededor de la semilla."""
    arr = io_zip.a_numpy(img)                # (z, y, x)
    i, j, k = seed                           # índice imagen (x, y, z)
    z, y, x = k, j, i
    sz = arr.shape
    zl, zh = max(0, z - radio), min(sz[0], z + radio + 1)
    yl, yh = max(0, y - radio), min(sz[1], y + radio + 1)
    xl, xh = max(0, x - radio), min(sz[2], x + radio + 1)
    parche = arr[zl:zh, yl:yh, xl:xh]
    return float(parche.mean()), float(parche.std())


def connected_threshold(img: sitk.Image, seed: Tuple[int, int, int],
                        alpha: float = 0.65, radio_cierre: int = 1) -> np.ndarray:
    """
    Region growing por VENTANA BRILLANTE: crece los vóxeles conexos a la semilla con
    intensidad en [alpha * I_semilla, max]. Como ET es la modalidad más brillante en
    T1c, restringir el crecimiento al tejido brillante evita que la región se "fugue"
    por los grises medios conectados de todo el cerebro (problema en volúmenes [0,1]).

    alpha controla cuán permisivo es el umbral inferior respecto a la intensidad de la
    semilla (más bajo -> región mayor; calibrado en ~0.65). `radio_cierre` cierra huecos.
    """
    I = float(img.GetPixel(tuple(int(s) for s in seed)))
    lo = alpha * I
    rg = sitk.ConnectedThreshold(img, seedList=[tuple(int(s) for s in seed)],
                                 lower=float(lo), upper=1.01)
    if radio_cierre > 0:
        rg = sitk.BinaryMorphologicalClosing(rg, [radio_cierre] * 3)
    return io_zip.a_numpy(sitk.Cast(rg, sitk.sitkUInt8)).astype(np.uint8)


def confidence_connected(img: sitk.Image, seed: Tuple[int, int, int],
                         multiplier: float = 2.5, iteraciones: int = 5,
                         radio_inicial: int = 2) -> np.ndarray:
    """Region growing por confianza (media +/- multiplier*sigma, re-estimado)."""
    rg = sitk.ConfidenceConnected(
        img, seedList=[tuple(int(s) for s in seed)],
        numberOfIterations=iteraciones, multiplier=multiplier,
        initialNeighborhoodRadius=radio_inicial, replaceValue=1)
    rg = sitk.BinaryMorphologicalClosing(rg, [1, 1, 1])
    return io_zip.a_numpy(sitk.Cast(rg, sitk.sitkUInt8)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Método 3 — Watershed marcador-controlado (de seg_watershed.py de Santiago)
# --------------------------------------------------------------------------- #
def segmentar_watershed(volumen: np.ndarray,
                        seed: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Watershed 3D marcador-controlado sobre la magnitud del gradiente de T1c.
    Marcador de fondo = tejido bajo (clase baja de Otsu); marcador de tumor = semilla
    (si se da) o clase alta de Otsu.
    """
    v = np.asarray(volumen, dtype=np.float32)
    cerebro = v != 0
    if cerebro.sum() < 10:
        return np.zeros_like(v, dtype=np.uint8)
    grad = sobel(v)
    marcadores = np.zeros(v.shape, dtype=np.int32)
    vals = v[cerebro]
    t = threshold_multiotsu(vals, classes=3)
    marcadores[(v > 0) & (v < t[0])] = 1                       # tejido bajo -> fondo
    if seed is not None:
        i, j, k = seed
        sm = np.zeros_like(marcadores, bool); sm[k, j, i] = True
        marcadores[ndi.binary_dilation(sm, iterations=2)] = 2  # semilla dilatada -> tumor
    else:
        marcadores[v > t[1]] = 2                               # clase alta -> tumor
    etiquetas = watershed(grad, markers=marcadores, mask=cerebro)
    return (etiquetas == 2).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Método 4 — K-means / GMM MULTIMODAL (de seg_kmeans_gmm.py de Santiago)
# --------------------------------------------------------------------------- #
def _matriz_features(volumenes: List[np.ndarray]):
    """Apila modalidades -> (N_vox_cerebro, n_mods) usando la unión de máscaras."""
    cerebro = np.zeros_like(volumenes[0], dtype=bool)
    for v in volumenes:
        cerebro |= (v != 0)
    X = np.stack([v[cerebro] for v in volumenes], axis=1).astype(np.float32)
    return X, cerebro


def segmentar_clustering(volumenes: List[np.ndarray], metodo: str = "gmm",
                         n_clusters: int = 4, idx_referencia: int = 0,
                         seed: int = C.SEED) -> np.ndarray:
    """
    Agrupamiento de intensidades MULTIMODAL (varias modalidades por vóxel). El cluster
    "tumor" es el de mayor intensidad media en la modalidad de referencia (T1c, idx 0),
    que aproxima el realce (ET). Devuelve máscara binaria uint8.
    """
    X, cerebro = _matriz_features(volumenes)
    if X.shape[0] < n_clusters:
        return np.zeros_like(volumenes[0], dtype=np.uint8)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xz = (X - mu) / sd
    if metodo == "kmeans":
        modelo = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        etiquetas = modelo.fit_predict(Xz)
    elif metodo == "gmm":
        modelo = GaussianMixture(n_components=n_clusters, covariance_type="full",
                                 random_state=seed)
        etiquetas = modelo.fit_predict(Xz)
    else:
        raise ValueError("metodo debe ser 'kmeans' o 'gmm'")
    medias = [X[etiquetas == c, idx_referencia].mean() if (etiquetas == c).any() else -np.inf
              for c in range(n_clusters)]
    cluster_tumor = int(np.argmax(medias))
    mask = np.zeros_like(volumenes[0], dtype=np.uint8)
    mask[cerebro] = (etiquetas == cluster_tumor).astype(np.uint8)
    return mask

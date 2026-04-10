# Taller de Segmentación por Umbrales con ITK

Segmentación por umbrales sobre imágenes de resonancia magnética (MR) de **tumor cerebral**,
**cáncer de mama** y **tumor hepático** utilizando ITK en Python.

Se implementan tres técnicas de segmentación:

| Técnica | Filtro ITK | Script |
|---------|-----------|--------|
| Umbral binario | `BinaryThresholdImageFilter` | `scripts/binary_threshold.py` |
| Otsu (múltiples umbrales) | `OtsuMultipleThresholdsImageFilter` | `scripts/otsu_segmentation.py` |
| K-Means | `ScalarImageKmeansImageFilter` | `scripts/kmeans_segmentation.py` |

## Imágenes de entrada

Las imágenes se encuentran en `../Images/`:

- `MRBrainTumor.nii.gz` — Resonancia magnética con tumor cerebral
- `MRBreastCancer.nii.gz` — Resonancia magnética con cáncer de mama
- `MRLiverTumor.nii.gz` — Resonancia magnética con tumor hepático

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Flujo de trabajo recomendado

Ejecutar todos los scripts desde la raíz del proyecto (`taller-segmentation/`):

### 1. Analizar histogramas de intensidad

```bash
python scripts/histogram.py
```

Genera histogramas individuales y comparativo en `report/figures/`.
Usar los percentiles y estadísticas impresas para ajustar los parámetros de segmentación.

### 2. Ajustar parámetros de umbral binario

Editar el diccionario `PARAMS` en `scripts/binary_threshold.py` según los valores
observados en el histograma.

### 3. Ejecutar segmentación por umbral binario

```bash
python scripts/binary_threshold.py
```

Resultados en `results/binary_threshold/`.

### 4. Ejecutar segmentación Otsu

```bash
python scripts/otsu_segmentation.py
```

Prueba automáticamente con 1, 2 y 3 umbrales. Resultados en `results/otsu/`.

### 5. Ejecutar segmentación K-Means

```bash
python scripts/kmeans_segmentation.py
```

Prueba automáticamente con 2, 3 y 4 clases. Resultados en `results/kmeans/`.

### 6. Visualizar resultados

```bash
python scripts/visualize_results.py
```

Editar las variables `IMAGE_PATH`, `SLICE_AXIAL`, `SLICE_CORONAL` y `SLICE_SAGITAL`
al inicio del script para explorar distintos volúmenes y cortes.

## Estructura de carpetas

```
taller-segmentation/
├── results/
│   ├── binary_threshold/   ← Resultados de umbral binario (.nii.gz)
│   ├── otsu/               ← Resultados de Otsu (.nii.gz)
│   └── kmeans/             ← Resultados de K-Means (.nii.gz)
├── scripts/
│   ├── histogram.py            ← Análisis de histogramas de intensidad
│   ├── binary_threshold.py     ← Segmentación por umbral binario
│   ├── otsu_segmentation.py    ← Segmentación Otsu (múltiples umbrales)
│   ├── kmeans_segmentation.py  ← Segmentación K-Means
│   └── visualize_results.py    ← Visualización de vistas ortogonales
├── report/
│   └── figures/            ← Figuras generadas (histogramas, vistas)
├── README.md
└── requirements.txt
```

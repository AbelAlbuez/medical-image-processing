# Taller 3 — Segmentación por Regiones

Implementación de algoritmos de segmentación basados en regiones usando ITK
sobre imágenes de Resonancia Magnética. Curso de Procesamiento de Imágenes
Médicas — Pontificia Universidad Javeriana, 2026.

Autores: Abel Albuez, Victoria Acero, Santiago Gil

---

## Algoritmos implementados

- **Watershed**: segmentación basada en la topografía del mapa de gradiente.
  Parámetros: sigma, threshold, level.
- **Level Sets con Fast Marching**: propagación de un frente de onda desde
  una semilla definida manualmente. Parámetros: iterations, sigma, alpha,
  beta, stoppingValue. Se ejecuta en dos fases.

---

## Imágenes médicas

| Imagen                | Voxel (x, y, z) | Intensidad | Spacing      |
|-----------------------|-----------------|------------|--------------|
| MRBrainTumor.nii.gz   | (140, 100, 83)  | 213        | Axial 1.2    |
| MRLiverTumor.nii.gz   | (28, 86, 73)    | 61         | Axial 1.7    |
| MRBreastCancer.nii.gz | (558, 250, 20)  | 390        | Reformat 0.4 |

Las imágenes deben colocarse manualmente en la carpeta `images/`.
No están incluidas en el repositorio (`.gitignore` las excluye).

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Estructura de carpetas

```
taller-segmentacion-second/
├── images/                        # Imágenes médicas (no incluidas en el repo)
├── output/
│   ├── watershed/
│   │   ├── volumes/               # Volúmenes .nii.gz por combinación
│   │   ├── individual/            # Figura 3 vistas por combinación
│   │   ├── grids/                 # Grillas comparativas
│   │   └── gradient/              # Imágenes de gradiente intermedias
│   ├── level_sets/
│   │   ├── fase1/
│   │   │   ├── volumes/
│   │   │   ├── individual/
│   │   │   └── grids/
│   │   └── fase2/
│   │       ├── volumes/
│   │       ├── individual/
│   │       └── grids/
│   └── mosaics/                   # Mosaicos resumen por imagen y método
├── report/
│   └── figures/
├── scripts/
│   ├── watershed.py               # Barrido de parámetros Watershed
│   ├── level_sets.py              # Barrido de parámetros Level Sets (2 fases)
│   ├── generate_mosaic.py         # Generador automático de mosaicos
│   └── visualize_results.py       # Visualizador individual
└── requirements.txt
```

---

## Flujo de trabajo recomendado

### Paso 1 — Watershed (no requiere semilla)

```bash
python scripts/watershed.py MRBrainTumor.nii.gz
python scripts/watershed.py MRLiverTumor.nii.gz
python scripts/watershed.py MRBreastCancer.nii.gz
```

### Paso 2 — Level Sets fase 1 (iterations + sigma)

```bash
python scripts/level_sets.py MRBrainTumor.nii.gz   --seed 140 100 83 --phase 1
python scripts/level_sets.py MRLiverTumor.nii.gz   --seed 28 86 73   --phase 1
python scripts/level_sets.py MRBreastCancer.nii.gz --seed 558 250 20 --phase 1
```

### Paso 3 — Generar mosaicos para revisar resultados

```bash
python scripts/generate_mosaic.py
```

### Paso 4 — Editar `ITERATIONS_BEST` y `SIGMA_BEST` en `level_sets.py`

Revisar los mosaicos de fase 1 en `output/mosaics/` y editar las constantes
al inicio de `scripts/level_sets.py`:

```python
ITERATIONS_BEST = <mejor valor de fase 1>
SIGMA_BEST      = <mejor valor de fase 1>
```

### Paso 5 — Level Sets fase 2 (alpha + beta + stoppingValue)

```bash
python scripts/level_sets.py MRBrainTumor.nii.gz   --seed 140 100 83 --phase 2
python scripts/level_sets.py MRLiverTumor.nii.gz   --seed 28 86 73   --phase 2
python scripts/level_sets.py MRBreastCancer.nii.gz --seed 558 250 20 --phase 2
```

### Paso 6 — Generar mosaicos finales

```bash
python scripts/generate_mosaic.py
```

---

## Advertencia de rendimiento

El paso más lento es la **difusión anisotrópica** en Level Sets.
Tiempos aproximados por combinación según `iterations`:

| iterations | Tiempo aproximado |
|------------|-------------------|
| 5          | ~30 segundos      |
| 15         | ~90 segundos      |
| 30         | ~3 minutos        |

La fase 1 completa (9 combinaciones) puede tardar entre **5 y 30 minutos**
dependiendo del hardware. Se imprime el tiempo por paso en consola para
identificar el cuello de botella.

---

## Salidas generadas

### Watershed

- `output/watershed/volumes/`    → volúmenes segmentados (`.nii.gz`)
- `output/watershed/individual/` → figura 3 vistas por combinación (`.png`)
- `output/watershed/grids/`      → grillas comparativas (`.png`)
- `output/watershed/gradient/`   → mapas de gradiente intermedios (`.nii.gz`)

### Level Sets

- `output/level_sets/fase1/` → resultados de la fase 1
- `output/level_sets/fase2/` → resultados de la fase 2
- Cada fase contiene: `volumes/`, `individual/`, `grids/`

### Mosaicos

- `output/mosaics/` → un mosaico por imagen y método (`.png`)

  Ejemplo: `MRBrainTumor_watershed_mosaic.png`

---

## Semillas identificadas en 3D Slicer

Las semillas para Level Sets se identificaron usando el **Data Probe** de 3D
Slicer, posicionando el cursor sobre la región del tumor en la vista axial.

| Imagen         | Voxel (x, y, z) | Intensidad | Spacing      |
|----------------|-----------------|------------|--------------|
| MRBrainTumor   | (140, 100, 83)  | 213        | Axial 1.2    |
| MRLiverTumor   | (28, 86, 73)    | 61         | Axial 1.7    |
| MRBreastCancer | (558, 250, 20)  | 390        | Reformat 0.4 |

> ⚠️ El spacing de MRBreastCancer corresponde a una vista reformateada
> en 3D Slicer, no al spacing axial nativo del volumen.

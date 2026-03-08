# Medical Image Processing — Class Filters

Proyecto de procesamiento de imágenes médicas con filtros (media, gradiente) usando ITK.

## Requisitos

- Python 3.x
- pip

## Configuración

### 1. Crear el entorno virtual (venv)

Desde la raíz del proyecto:

```bash
python3 -m venv venv
```

### 2. Activar el venv

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows (cmd):**

```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

Con el venv activado:

```bash
pip install -r requirenments.txt
```

### 4. Descargar imágenes de muestra (samples)

Las imágenes de ejemplo (p. ej. `CTACardio.nii`, `MRBrainTumor1_3.nii`, `MRHead.nii.gz`, `USProstate_1.nii`) no están en el repositorio por su tamaño. Debes descargarlas desde **3D Slicer** (Sample Data) y colocarlas en la carpeta `samples/` de este proyecto para poder ejecutar los ejemplos del readme.

## Ejecutar el proyecto

### Filtro de media (`mean.py`)

Aplica un filtro de media (suavizado) a una imagen 3D. Argumentos: imagen de entrada, imagen de salida, radio (entero, en píxeles).

```bash
python ./src/mean.py <input_image> <output_image> <radius>
```

**Ejemplo con imagen de muestra:**

```bash
python ./src/mean.py samples/CTACardio.nii ./result/sample.nii 3
```

### Filtro de gradiente (`gradient.py`)

Calcula la magnitud del gradiente (imagen 3D). Argumentos: imagen de entrada, imagen de salida.

```bash
python ./src/gradient.py <input_image> <output_image>
```

**Ejemplo:**

```bash
python ./src/gradient.py samples/some_slice.png ./result/gradient_output.nii
```

### Ejecutar todos los filtros (experimentos) — `run_all_filters.py`

El script **`run_all_filters.py`** es un *runner* que ejecuta automáticamente los filtros de `src/` sobre todas las imágenes en `samples/`, sin modificar la lógica de los scripts existentes (los invoca por línea de comandos con `subprocess`).

**Qué hace:**

- Recorre todos los archivos en `samples/` (extensiones `.nii` y `.nii.gz`).
- Para cada imagen, ejecuta:
  - **Gradient** (sin parámetros) → una salida por imagen.
  - **Mean** con radios 2, 3 y 4 → tres salidas por imagen.
  - **Median** con radios 2, 3 y 4 → tres salidas por imagen.
  - **Adaptive histogram** (histogram.py) con varias combinaciones de alpha, beta y radius → varias salidas por imagen.
- Guarda los resultados en carpetas separadas bajo `result/`:
  - `result/gradient_results/`
  - `result/mean_results/`
  - `result/median_results/`
  - `result/adaptive_histogram_results/`
- Los nombres de salida siguen un patrón claro, por ejemplo:
  - `CTACardio_gradient.nii`, `MRHead_mean_r3.nii`, `USProstate_1_hist_a0.5_b0.5_r3.nii`.
- Crea las carpetas si no existen, escribe logs en consola y, si un experimento falla, continúa con el resto y al final muestra un resumen de éxitos y fallos.
- **Comparaciones PNG:** Tras ejecutar los filtros, genera figuras de comparación (corte axial central) en `comparison_results/`: **mean** (Original | r2 | r3 | r4), **median** (Original | r2 | r3 | r4), **gradient** (Original | Gradient), **adaptive_histogram** (Original | algunas combinaciones alpha/beta/radius). Ejemplo: `comparison_results/median/CTACardio_comparison.png`.

**Cómo ejecutarlo** (con el venv activado, desde la raíz del proyecto):

```bash
python run_all_filters.py
```

El análisis detallado de cada filtro y sus parámetros está en **`FILTER_ANALYSIS.md`**. Todos los filtros están configurados para imágenes 3D (volúmenes NIfTI).

## Estructura del proyecto

```
class-filter-image/
├── readme.md
├── requirenments.txt
├── run_all_filters.py       # runner de experimentos (ejecuta todos los filtros sobre samples/)
├── FILTER_ANALYSIS.md       # análisis de filtros ITK y propuesta de experimentación
├── venv/                    # entorno virtual (no versionar)
├── src/
│   ├── mean.py              # filtro de media (3D)
│   ├── median.py            # filtro de mediana (3D)
│   ├── gradient.py          # filtro de magnitud del gradiente (3D)
│   ├── histogram.py         # equalización adaptativa del histograma (3D)
│   └── histogram_plot.py    # visualización del histograma
├── samples/                 # imágenes de ejemplo (descargar desde 3D Slicer)
│   ├── CTACardio.nii
│   ├── MRBrainTumor1_3.nii
│   ├── MRHead.nii.gz
│   └── USProstate_1.nii
├── modalidades/             # ejemplos por modalidad (ITK)
│   ├── AdaptiveHistogramEqualizationImageFilter/
│   ├── ComputeGradientMagnitude/
│   ├── MeanFilteringOfAnImage/
│   └── MedianFilteringOfAnImage/
├── result/                  # imágenes de salida
│   ├── gradient_results/
│   ├── mean_results/
│   ├── median_results/
│   └── adaptive_histogram_results/
├── comparison_results/      # PNG de comparación por filtro
│   ├── mean/
│   ├── median/
│   ├── gradient/
│   └── adaptive_histogram/
└── comparison_visualizer.py  # generación de figuras de comparación
```

## Dependencias principales

- **itk** — Insight Toolkit para imágenes médicas
- **numpy** — arrays numéricos
- **matplotlib** — visualización
- **jupyter** — notebooks

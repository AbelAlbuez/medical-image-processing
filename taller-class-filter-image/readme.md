# Taller 1 — Filtros de Denoising en Imágenes MRI

**Autores:** Abel Albuez · Victoria Acero · Santiago Gil
**Institución:** Pontificia Universidad Javeriana
**Curso:** Procesamiento de Imágenes Médicas
**Repositorio:** https://github.com/AbelAlbuez/medical-image-processing

---

## Resumen

Implementación y evaluación de tres filtros clásicos de denoising aplicados a volúmenes
cerebrales T1 del simulador BrainWeb (pn3, pn5, pn9; inhomogeneidad RF 20%), siguiendo
la taxonomía de Ali (2018):

| Sección | Filtro | Tipo |
|---------|--------|------|
| 2.2.1 | Filtro de Mediana (MF) | Espacial, no lineal |
| 2.2.2 | Filtro de Mediana Adaptativo (AMF) | Espacial, no lineal, ventana dinámica |
| 2.1   | Filtro de Wiener Adaptativo (AWF) | Espacial, estadístico (MMSE) |

Los tres filtros se implementaron en Python con **ITK** para la lectura/escritura de
volúmenes NIfTI y **NumPy/SciPy** para la lógica de filtrado. Los resultados se
documentan en el informe LaTeX compilado (`taller1_filtrado.pdf`).

---

## Requisitos

- Python 3.x
- pip
- Dependencias (ver `requirenments.txt`): `itk>=5.4.5`, `numpy`, `matplotlib`, `scipy`
- Opcional: `tqdm` (barra de progreso en el modo `--experiment`)

---

## Configuración del entorno virtual

```bash
# Crear el entorno virtual
python -m venv venv
```

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

**Instalar dependencias:**
```bash
pip install -r requirenments.txt
# opcional
pip install tqdm
```

---

## Imágenes de prueba

Coloca los volúmenes NIfTI (`.nii` o `.nii.gz`) en la carpeta **`Images/`**.
Todos los scripts aceptan tanto rutas absolutas como solo el nombre del archivo
(se busca automáticamente dentro de `Images/`).

Los volúmenes usados en el taller son los del simulador **BrainWeb**:
- `t1_icbm_normal_1mm_pn3_rf20.nii.gz` — ruido 3% (bajo)
- `t1_icbm_normal_1mm_pn5_rf20.nii.gz` — ruido 5% (medio)
- `t1_icbm_normal_1mm_pn9_rf20.nii.gz` — ruido 9% (alto)

> **Nota sobre rutas no-ASCII en Windows:** el proyecto reside en una carpeta con
> tildes (`imágenes médicas`). La capa C++ de ITK no puede abrir rutas con caracteres
> no-ASCII en Windows. Todos los scripts resuelven esto copiando el archivo a un
> directorio temporal ASCII antes de procesarlo.

---

## Scripts de filtrado

### `src/median.py` — Filtro de Mediana

Implementa el algoritmo descrito en Ali 2018 (sección 2.2.1): para cada vóxel, reemplaza
su valor por la **mediana de los 26 vecinos** de la vecindad cúbica (3×3×3), excluyendo
el vóxel central. La lógica es equivalente al bucle explícito de referencia:

```python
for z in range(1, dim_z - 1):
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            vecindad           = volumen[z-1:z+2, y-1:y+2, x-1:x+2]
            vecinos            = vecindad.flatten()
            vecinos_sin_centro = np.delete(vecinos, 13)
            salida[z, y, x]   = np.median(vecinos_sin_centro)
```

Internamente se vectoriza con `sliding_window_view` para mayor velocidad, pero el
resultado es idéntico. Lee y escribe los volúmenes como **float32** (`itk.F`).

```bash
python src/median.py <entrada> <radio>
python src/median.py <entrada> <radio> <salida.nii>

# Ejemplos:
python src/median.py t1_icbm_normal_1mm_pn5_rf20.nii.gz 1
python src/median.py t1_icbm_normal_1mm_pn5_rf20.nii.gz 1 output/median/resultado.nii
```

| Argumento | Descripción |
|-----------|-------------|
| `input_image` | Ruta al volumen de entrada (o nombre dentro de `Images/`) |
| `radius` | Radio de la ventana cúbica: `1` → ventana 3×3×3 |
| `output_image` | Opcional; por defecto `output/median/<base>_median_r{radio}.nii` |

---

### `src/adaptive-median.py` — Filtro de Mediana Adaptativo

Implementa el algoritmo de dos niveles de Ali 2018 (sección 2.2.2) con ventana dinámica
que crece de 3 hasta `Smax`. Ofrece dos variantes:

- **Variante ITK** (por defecto, más rápida): MedianImageFilter + GradientMagnitude + umbral.
- **Variante NumPy** (`--no-itk`): implementación exacta del artículo, píxel a píxel.

```bash
# Uso básico (variante ITK)
python src/adaptive-median.py <entrada>
python src/adaptive-median.py <entrada> <salida.nii> --max-window 7

# Variante NumPy exacta (más lenta)
python src/adaptive-median.py <entrada> --no-itk

# Con inyección de ruido antes de filtrar
python src/adaptive-median.py <entrada> --noise-type salt_pepper --noise-density 0.1
python src/adaptive-median.py <entrada> --noise-type gaussian --noise-sigma 10.0
python src/adaptive-median.py <entrada> --noise-type mixed

# Modo experimento: 12 combinaciones predefinidas + figuras resumen
python src/adaptive-median.py <entrada> --experiment
```

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `input_image` | — | Ruta o nombre en `Images/` |
| `output_image` | automático | `output/adaptive-median/<base>_adaptive_median.nii` |
| `--max-window` | 7 | Smax: tamaño máximo de ventana (impar) |
| `--no-itk` | False | Usa la implementación NumPy exacta |
| `--noise-type` | `none` | `none` \| `salt_pepper` \| `gaussian` \| `mixed` |
| `--noise-density` | 0.1 | Fracción de píxeles afectados (sal y pimienta) |
| `--noise-sigma` | 10.0 | Desviación estándar del ruido gaussiano |
| `--experiment` | False | Ejecuta las 12 combinaciones predefinidas |

---

### `src/wiener.py` — Filtro de Wiener Adaptativo

Implementa el estimador MMSE de Ali 2018 (sección 2.1). Opera completamente en 3D
mediante `scipy.ndimage.uniform_filter` para el cálculo vectorizado de media y varianza
locales. La varianza global del ruido se estima automáticamente como promedio de todas
las varianzas locales.

Fórmula por vóxel:
```
f_hat = mu + max(0, var_local - nu2) / max(var_local, eps) * (g - mu)
```

```bash
python src/wiener.py <entrada>
python src/wiener.py <entrada> <salida.nii> --window 5

# Con ruido sintético
python src/wiener.py <entrada> --noise-type gaussian --noise-sigma 10.0

# Con imagen de referencia para calcular PSNR
python src/wiener.py <entrada_ruidosa.nii.gz> --reference <referencia_limpia.nii.gz>
```

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `input_image` | — | Ruta o nombre en `Images/` |
| `output_image` | automático | `output/wiener/<base>_wiener.nii` |
| `--window` | 5 | Lado de la ventana cúbica N×N×N (impar) |
| `--noise-var` | automático | Varianza de ruido nu2 (si no se indica, se estima) |
| `--noise-type` | `none` | `none` \| `salt_pepper` \| `gaussian` \| `mixed` |
| `--noise-density` | 0.1 | Fracción de píxeles sal y pimienta |
| `--noise-sigma` | 10.0 | Desviación estándar del ruido gaussiano |
| `--reference` | — | Imagen limpia de referencia para PSNR (opcional) |

---

## Scripts de soporte

### `src/run_all.py` — Ejecutor por lotes

Aplica los tres filtros sobre las tres imágenes BrainWeb (pn3, pn5, pn9) y genera
los PNG comparativos del informe. Es el script principal para reproducir todos los
resultados del taller en una sola ejecución.

```bash
python src/run_all.py
python src/run_all.py --median-radius 1 --max-window 7 --wiener-window 5
python src/run_all.py --force   # re-ejecuta aunque las salidas ya existan
```

Genera en `output/report/`:
- `<base>_comparison.png` — 4 columnas (Original, Mediana, Med. Adaptativa, Wiener) × 3 planos
- `summary_axial.png` — mosaico resumen: todos los filtros × todos los niveles de ruido

### `src/comparison_median.py` — Comparación visual Mediana vs. Mediana Adaptativa

Panel PNG de 3 columnas (Original | Mediana | Mediana Adaptativa) para una sola imagen.

```bash
python src/comparison_median.py <entrada> [--radius 3] [--max-window 7] [--no-itk]
```

Guarda en `output/comparison_results/<base>_comparison.png`.

### `src/metrics.py` — Métricas cuantitativas (PSNR / MSE)

Replica las tablas de comparación del artículo de Ali (2018): inyecta ruido sintético
sobre una imagen de referencia y mide el PSNR y MSE de cada filtro.

```bash
# Tabla gaussiana
python src/metrics.py t1_icbm_normal_1mm_pn3_rf20.nii.gz --mode gaussian

# Tabla sal y pimienta
python src/metrics.py t1_icbm_normal_1mm_pn3_rf20.nii.gz --mode salt_pepper

# Ambas + CSV + gráfico de barras
python src/metrics.py t1_icbm_normal_1mm_pn3_rf20.nii.gz --mode both --save-csv --save-png
```

---

## Estructura de salidas

```
output/
├── median/                  # Volúmenes NIfTI del filtro de mediana
├── adaptive-median/         # Volúmenes NIfTI del filtro de mediana adaptativa
├── wiener/                  # Volúmenes NIfTI del filtro de Wiener
├── comparison_results/      # PNGs individuales por filtro y experimentos
├── report/                  # PNGs de comparación completa + summary_axial.png
└── metrics/                 # CSV y gráficos de barras de PSNR/MSE (si se generan)
```

---

## Informe del taller

El documento completo se encuentra en:
- **Fuente LaTeX:** `taller1_filtrado.tex`
- **PDF compilado:** `taller1_filtrado.pdf` (23 páginas)

Compilar manualmente (requiere MiKTeX o TeX Live):
```bash
pdflatex -interaction=nonstopmode taller1_filtrado.tex
pdflatex -interaction=nonstopmode taller1_filtrado.tex  # segunda pasada para referencias
```

---

## Estructura del proyecto

```
taller-class-filter-image/
├── readme.md
├── requirenments.txt
├── taller1_filtrado.tex       # Informe LaTeX
├── taller1_filtrado.pdf       # Informe compilado (23 páginas)
├── Images/                    # Volúmenes NIfTI BrainWeb de entrada
├── src/
│   ├── median.py              # Filtro de Mediana (Ali 2018, sec 2.2.1)
│   ├── adaptive-median.py     # Filtro de Mediana Adaptativa (Ali 2018, sec 2.2.2)
│   ├── wiener.py              # Filtro de Wiener Adaptativo (Ali 2018, sec 2.1)
│   ├── run_all.py             # Ejecutor por lotes (genera todos los PNG del informe)
│   ├── comparison_median.py   # Comparación visual Mediana vs. Mediana Adaptativa
│   └── metrics.py             # Métricas cuantitativas PSNR/MSE
├── output/                    # Generado al ejecutar los scripts
└── venv/                      # Entorno virtual (no versionado)
```

---

## Tecnologías

- **Python** 3.x
- **ITK** — lectura/escritura de volúmenes NIfTI y filtros nativos
- **NumPy** — implementación de algoritmos de filtrado
- **SciPy** — `uniform_filter` y `median_filter` para filtros vectorizados
- **Matplotlib** — generación de PNGs comparativos
- **LaTeX / MiKTeX** — informe del taller

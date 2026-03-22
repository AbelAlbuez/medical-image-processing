# Procesamiento de imágenes médicas — Filtros clásicos (taller)

**Autores:** Abel Albuez, Victoria Acero, Santiago Gil  
**Institución:** Pontificia Universidad Javeriana  
**Curso:** Procesamiento de imágenes médicas  

**Repositorio:** https://github.com/AbelAlbuez/medical-image-processing

## Resumen

Proyecto académico sobre volúmenes 3D NIfTI con **ITK** y **NumPy**: filtro de **mediana** clásico, **mediana adaptativa** (aproximación ITK o implementación NumPy con `--no-itk`), **inyección de ruido** (sal y pimienta, Gaussiano, mixto), modo **`--experiment`** con varias combinaciones y figuras PNG de comparación. Enfoque en procesamiento clásico (sin aprendizaje profundo). El script **`comparison_median.py`** genera un panel Original / mediana / adaptativa.

## Requisitos

- Python 3.x  
- pip  
- **Dependencias:** `itk>=5.4.5`, `numpy`, `matplotlib` (ver `requirenments.txt`)  
- Opcional: **`tqdm`** (barra de progreso en `--experiment`)

## Configuración

**Crear el entorno virtual**

```bash
python3 -m venv venv
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

**Instalar dependencias** (el archivo se llama `requirenments.txt`, con la ortografía indicada):

```bash
pip install -r requirenments.txt
```

```bash
# opcional
pip install tqdm
```

## Imágenes de prueba

Coloca los volúmenes NIfTI (`.nii` o `.nii.gz`) en la carpeta **`Images/`**. Los scripts resuelven también por nombre de archivo dentro de `Images/` si no pasas una ruta absoluta. Puedes usar datos de **3D Slicer → Sample Data** u otros volúmenes compatibles.

## Uso

### `median.py` (MedianImageFilter, 3D)

Orden de argumentos: entrada, radio, salida opcional. Si omites la salida o no es una ruta absoluta, se guarda en `output/median/<base>_median_r{radius}.nii`.

```bash
python src/median.py <entrada> <radius>
python src/median.py <entrada> <radius> <salida_absoluta.nii>
```

### `adaptive-median.py` (mediana adaptativa, sección 2.2.2)

Nombre del archivo: **`adaptive-median.py`** (con guion).

```bash
# salida por defecto en output/adaptive-median/
python src/adaptive-median.py <entrada>

python src/adaptive-median.py <entrada> <salida.nii> --max-window 7

# ruido + filtro
python src/adaptive-median.py <entrada> --noise-type salt_pepper --noise-density 0.1 --max-window 7

# implementación NumPy (más lenta en volúmenes grandes)
python src/adaptive-median.py <entrada> --no-itk

# las 12 combinaciones del experimento + PNG resumen (recomendado ITK, sin --no-itk)
python src/adaptive-median.py <entrada> --experiment
```

| Argumento | Descripción |
|-----------|-------------|
| `input_image` | Posicional; también se busca en `Images/` |
| `output_image` | Opcional; por defecto `output/adaptive-median/<base>_adaptive_median.nii` |
| `--max-window` | `Smax` (impar; por defecto 7) |
| `--no-itk` | Usa NumPy en lugar de la aproximación ITK |
| `--noise-type` | `none`, `salt_pepper`, `gaussian`, `mixed` |
| `--noise-density` | Fracción sal/pimienta (por defecto 0.1) |
| `--noise-sigma` | σ Gaussiano (por defecto 10.0) |
| `--experiment` | Ejecuta todas las combinaciones predefinidas y genera PNG individuales + resumen |

### `comparison_median.py`

```bash
python src/comparison_median.py <entrada> [--radius 3] [--max-window 7] [--no-itk]
```

Genera `output/comparison_results/<base>_comparison.png` (Original | mediana | adaptativa).

### Wiener adaptativo

**🚧 En desarrollo** — el script `wiener.py` aún no está en este repositorio.

## Salidas

Las carpetas se crean al ejecutar los scripts:

```
output/
├── median/                 # resultados de median.py
├── adaptive-median/        # resultados de adaptive-median.py
└── comparison_results/     # PNG de adaptive-median (incl. modo --experiment) y comparison_median.py
```

Con **`--experiment`**: varios `*_adaptive_median.nii` en `output/adaptive-median/`, PNG por combinación y `*_experiment_summary.png` en `output/comparison_results/`.

## Filtros y scripts

| Filtro / herramienta | Script | Parámetros (resumen) | Descripción |
|----------------------|--------|----------------------|-------------|
| **Mediana** | `src/median.py` | `input_image`, `radius`, `output_image` opcional | `MedianImageFilter` ITK 3D. Mayor radio → más suavizado y menos ruido tipo sal y pimienta. |
| **Mediana adaptativa** | `src/adaptive-median.py` | Ver tabla de argumentos arriba | Aproximación ITK (mediana en plano + preservación de bordes por gradiente) o algoritmo NumPy (Ali 2018); ruido sintético y modo experimento. |
| **Comparación visual** | `src/comparison_median.py` | `input_image`, `--radius`, `--max-window`, `--no-itk` | Panel de tres columnas para informes. |
| **Wiener adaptativo** | `wiener.py` | — | **🚧 Pendiente** — no incluido aún. |

## Estructura del proyecto

```
taller-class-filter-image/
├── readme.md
├── requirenments.txt
├── .gitignore
├── Images/                    # volúmenes NIfTI de prueba
├── src/
│   ├── median.py
│   ├── adaptive-median.py
│   └── comparison_median.py
├── output/                    # generado al ejecutar (median, adaptive-median, comparison_results)
└── venv/
```

## Tecnologías

- **Python** 3.x · **ITK** · **NumPy** · **Matplotlib**  
- **Documento del taller:** `taller_filtros_mediana.docx` (en el repositorio raíz del curso, si está versionado)

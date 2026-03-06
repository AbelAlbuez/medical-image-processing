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

Calcula la magnitud del gradiente (imagen 2D). Argumentos: imagen de entrada, imagen de salida.

```bash
python ./src/gradient.py <input_image> <output_image>
```

**Ejemplo:**

```bash
python ./src/gradient.py samples/some_slice.png ./result/gradient_output.nii
```

## Estructura del proyecto

```
class-filter-image/
├── readme.md
├── requirenments.txt
├── venv/                    # entorno virtual (no versionar)
├── src/
│   ├── mean.py              # filtro de media (3D)
│   ├── median.py            # filtro de mediana
│   ├── gradient.py          # filtro de magnitud del gradiente (2D)
│   ├── histogram.py         # histograma de imagen
│   └── histogram_plot.py    # visualización del histograma
├── samples/                 # imágenes de ejemplo
│   ├── CTACardio.nii
│   ├── MRBrainTumor1_3.nii
│   ├── MRHead.nii.gz
│   └── USProstate_1.nii
├── modalidades/             # ejemplos por modalidad (ITK)
│   ├── AdaptiveHistogramEqualizationImageFilter/
│   ├── ComputeGradientMagnitude/
│   ├── MeanFilteringOfAnImage/
│   └── MedianFilteringOfAnImage/
└── result/                  # imágenes de salida
    └── sample.nii
```

## Dependencias principales

- **itk** — Insight Toolkit para imágenes médicas
- **numpy** — arrays numéricos
- **matplotlib** — visualización
- **jupyter** — notebooks

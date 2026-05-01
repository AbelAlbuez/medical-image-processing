# taller-segmentacion-second

Taller 3 - Segmentacion por regiones con ITK: **Watersheds** y **Level Sets (Fast Marching)**.

## Estructura

```
taller-segmentacion-second/
├── images/                       # imagenes de entrada (.nii.gz)
├── output/
│   ├── watershed/
│   │   └── gradient/             # imagenes de gradiente intermedias
│   └── level_sets/
│       └── gradient/             # imagenes de gradiente intermedias
├── report/
│   └── figures/                  # figuras del informe
├── scripts/
│   ├── watershed.py              # pipeline Watersheds
│   ├── level_sets.py             # pipeline Fast Marching
│   └── visualize_results.py      # vista 3 planos de un volumen
├── requirements.txt
└── README.md
```

## Dependencias (`requirements.txt`)

```
itk==5.4.0
itk-io==5.4.0
itk-filtering==5.4.0
itk-segmentation==5.4.0
numpy==1.26.4
matplotlib==3.8.4
```

## Setup: crear entorno virtual e instalar dependencias

Desde el directorio `taller-segmentacion-second/`:

1. Crear el entorno virtual (solo la primera vez):

   ```bash
   python3 -m venv venv
   ```

2. Activar el entorno virtual:

   ```bash
   source venv/bin/activate
   ```

3. Actualizar `pip` e instalar dependencias:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Salir del entorno virtual cuando termines:

   ```bash
   deactivate
   ```

## Uso

### Imagenes de entrada

Copiar manualmente los volumenes `.nii.gz` dentro de la carpeta `images/`. Los scripts
resuelven rutas relativas contra esa carpeta automaticamente.

### Watershed - `scripts/watershed.py`

Pipeline:
1. Lectura de la imagen 3D float
2. `GradientMagnitudeRecursiveGaussianImageFilter` (parametro: `sigma`)
3. `WatershedImageFilter` (parametros: `threshold`, `level`)
4. Guardado del label map y de la imagen de gradiente intermedia

Ejecucion:

```bash
python scripts/watershed.py A1_grayT1.nii.gz \
    --sigma 1.0 \
    --threshold 0.01 \
    --level 0.2
```

Argumentos:

| Argumento     | Default              | Descripcion |
|---------------|----------------------|-------------|
| `input_image` | -                    | Ruta al `.nii.gz` (posicional). Si es relativa, se busca en `images/`. |
| `--sigma`     | `1.0`                | Sigma del gradiente gaussiano. |
| `--threshold` | `0.01`               | Umbral minimo de gradiente considerado borde. |
| `--level`     | `0.2`                | Nivel de inundacion (fusion de cuencas). |
| `--output_dir`| `output/watershed/`  | Directorio de salida. |

Salida:
- `output/watershed/<nombre>_ws_s{sigma}_t{threshold}_l{level}.nii.gz` - label map
- `output/watershed/gradient/<nombre>_gradient.nii.gz` - imagen de gradiente
- `output/watershed/<nombre>_ws_preview.png` - vista rapida 3 planos
- En consola: estadisticas del gradiente (min, max, media) y tiempo total.

### Level Sets - `scripts/level_sets.py`

Pipeline (hasta seccion 4.3.1 del ITK Software Guide - Fast Marching):
1. Lectura de la imagen 3D float
2. `CurvatureAnisotropicDiffusionImageFilter` (cuello de botella computacional)
3. `GradientMagnitudeRecursiveGaussianImageFilter` (parametro: `sigma`)
4. `SigmoidImageFilter` (parametros: `alpha`, `beta`) - genera la speed image
5. Creacion de la semilla con `initialDistance`
6. `FastMarchingImageFilter` (parametro: `stoppingValue`)
7. `BinaryThresholdImageFilter` - mascara binaria final

Ejecucion:

```bash
python scripts/level_sets.py A1_grayT1.nii.gz \
    --seed 132 142 96 \
    --initial_distance 5.0 \
    --sigma 1.0 \
    --alpha -0.5 \
    --beta 3.0 \
    --stopping_value 100.0 \
    --iterations 5
```

Argumentos:

| Argumento            | Default               | Descripcion |
|----------------------|-----------------------|-------------|
| `input_image`        | -                     | Ruta al `.nii.gz` (posicional). |
| `--seed X Y Z`       | centro del volumen    | Semilla en indices de voxel. |
| `--initial_distance` | `5.0`                 | Distancia inicial desde la semilla. |
| `--sigma`            | `1.0`                 | Sigma del gradiente gaussiano. |
| `--alpha`            | `-0.5`                | Pendiente del sigmoide. |
| `--beta`             | `3.0`                 | Punto de inflexion del sigmoide. |
| `--stopping_value`   | `100.0`               | Valor de parada del Fast Marching. |
| `--iterations`       | `5`                   | Iteraciones de difusion anisotropica. |
| `--time_step`        | `0.0625`              | Paso de tiempo de la difusion. |
| `--conductance`      | `3.0`                 | Conductancia de la difusion. |
| `--output_dir`       | `output/level_sets/`  | Directorio de salida. |

Salida:
- `output/level_sets/<nombre>_ls_seed{x}_{y}_{z}_iter{n}.nii.gz` - mascara binaria
- `output/level_sets/gradient/<nombre>_gradient.nii.gz` - imagen de gradiente
- En consola: estadisticas del gradiente y tabla de tiempos por paso.

### Visualizacion - `scripts/visualize_results.py`

Genera una figura con 3 cortes ortogonales (axial, coronal, sagital) de un volumen.
Editar las variables al inicio del script:

```python
IMAGE_PATH         = "output/watershed/<archivo>.nii.gz"
SLICE_AXIAL        = None    # None -> centro del eje
SLICE_CORONAL      = None
SLICE_SAGITAL      = None
USE_LABEL_COLORMAP = True    # True para segmentaciones, False para originales
```

Ejecucion:

```bash
python scripts/visualize_results.py
```

Notas:
- `USE_LABEL_COLORMAP=True` aplica `nipy_spectral`, esencial para distinguir las
  cuencas adyacentes producidas por Watershed (en escala de grises serian
  indistinguibles).
- `USE_LABEL_COLORMAP=False` aplica `gray` y se recomienda para los volumenes
  originales o las imagenes de gradiente.

## Referencias

- ITK Software Guide - Watersheds: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1860004.2
- ITK Software Guide - Fast Marching: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch4.html#x35-1890004.3

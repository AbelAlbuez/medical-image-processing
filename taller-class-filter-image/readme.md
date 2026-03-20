# Second class — filtros de imagen (taller)

Base del taller: scripts en `src/` para implementar filtros ITK sobre volúmenes 3D.

## Datos

Las imágenes de prueba están en **`Images/`** (archivos `.nii.gz`).

## Configuración

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirenments.txt
```

## Scripts en `src/`

| Archivo | Filtro objetivo (ITK) | Argumentos CLI |
|---------|------------------------|----------------|
| `mean.py` | `MeanImageFilter` | `input_image` `output_image` `radius` |
| `median.py` | `MedianImageFilter` | `input_image` `output_image` `radius` |
| `gradient.py` | `GradientMagnitudeImageFilter` | `input_image` `output_image` |
| `histogram.py` | `AdaptiveHistogramEqualizationImageFilter` | `input_image` `output_image` `alpha` `beta` `radius` |
| `histogram_plot.py` | (matplotlib, opcional) | no es filtro ITK; plantilla vacía para gráficos |

Cada plantilla incluye comentarios `TALLER:` donde debes conectar el filtro entre el lector y el escritor. Hasta implementar el filtro, los scripts escriben la imagen de entrada sin modificar (passthrough) para validar lectura/escritura.

## Ejemplo (tras implementar el filtro)

```bash
python src/mean.py Images/t1_icbm_normal_1mm_pn1_rf20.nii.gz result/mean_out.nii 3
```

Crea la carpeta `result/` si no existe.

## Referencia

La estructura sigue el proyecto `class-filter-image` del mismo repositorio.

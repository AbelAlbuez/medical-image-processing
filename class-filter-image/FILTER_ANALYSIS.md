# Análisis de filtros ITK — Proyecto class-filter-image

Documento de análisis para la base de experimentación. No se modifica la lógica de los scripts en `src`.

---

## 1. Análisis por archivo

### Archivo: `src/mean.py`

| Campo | Valor |
|-------|--------|
| **Filtro ITK** | `MeanImageFilter` |
| **Argumentos actuales** | `input_image`, `output_image`, `radius` (int) |
| **Parámetros experimentales** | `radius` (obligatorio en CLI; es el único que afecta el resultado) |
| **Cómo afectan el resultado** | Mayor `radius` → mayor suavizado (vecindad más grande). Valores típicos 1–5 píxeles. |

- **Dimensión:** 3D  
- **Tipo de pixel:** `itk.UC` (unsigned char)  
- **Salida:** imagen en disco (path indicado por `output_image`).

---

### Archivo: `src/median.py`

| Campo | Valor |
|-------|--------|
| **Filtro ITK** | `MedianImageFilter` |
| **Argumentos actuales** | `input_image`, `output_image`, `radius` (int) |
| **Parámetros experimentales** | `radius` (obligatorio en CLI; es el único que afecta el resultado) |
| **Cómo afectan el resultado** | Mayor `radius` → mayor suavizado y mayor reducción de ruido tipo sal y pimienta. Valores típicos 1–5. |

- **Dimensión:** 3D  
- **Tipo de pixel:** `itk.UC` (unsigned char)  
- **Salida:** imagen en disco.

---

### Archivo: `src/gradient.py`

| Campo | Valor |
|-------|--------|
| **Filtro ITK** | `GradientMagnitudeImageFilter` |
| **Argumentos actuales** | `input_image`, `output_image` |
| **Parámetros experimentales** | Ninguno. No tiene parámetros configurables en este script. |
| **Cómo afectan el resultado** | N/A — salida es magnitud del gradiente (bordes). |

- **Dimensión:** 3D  
- **Tipo de pixel:** entrada `itk.UC`, salida `itk.F` (float)  
- **Salida:** imagen en disco.

---

### Archivo: `src/histogram.py`

| Campo | Valor |
|-------|--------|
| **Filtro ITK** | `AdaptiveHistogramEqualizationImageFilter` |
| **Argumentos actuales** | `input_image`, `output_image`, `alpha` (float), `beta` (float), `radius` (int) |
| **Parámetros experimentales** | `alpha`, `beta`, `radius` (todos obligatorios en CLI; afectan contraste local) |
| **Cómo afectan el resultado** | `alpha`: control de contraste en zonas de pendiente pronunciada (ej. 0.3–1.0). `beta`: control en zonas planas (ej. 0.3–1.0). `radius`: tamaño de la vecindad para la equalización (ej. 3–7). |

- **Dimensión:** 3D  
- **Tipo de pixel:** `unsigned char`  
- **Salida:** imagen en disco vía `itk.imwrite`.

---

### Archivo: `src/histogram_plot.py`

| Campo | Valor |
|-------|--------|
| **Filtro ITK** | Ninguno (no es un filtro ITK) |
| **Argumentos actuales** | `sys.argv[1]`: ruta de imagen; `sys.argv[2]`: título del gráfico |
| **Uso** | Lee imagen con matplotlib (formato típico 2D: PNG/JPEG), dibuja histograma, guarda PNG. No procesa NIfTI ni forma parte de la cadena de filtros ITK. |

- **No incluido** en el runner de experimentos de filtros (solo gradient, mean, median, histogram).

---

## 2. Diferencias importantes entre scripts

| Aspecto | mean.py | median.py | gradient.py | histogram.py |
|---------|---------|------------|-------------|--------------|
| **Dimensión** | 3D | 3D | 3D | 3D |
| **Pixel entrada** | UC | UC | UC | unsigned char |
| **Pixel salida** | UC | UC | float | mismo |
| **Limitación** | Solo imágenes 3D | Solo imágenes 3D | Solo imágenes 3D | Solo imágenes 3D |

- **Consecuencia:** Todos los filtros están configurados para imágenes 3D (volúmenes NIfTI). Las imágenes en `samples/` son compatibles con los cuatro scripts.

---

## 3. Propuesta de experimentación

| Filtro | Parámetro | Valores sugeridos | Efecto esperado |
|--------|-----------|-------------------|-----------------|
| **AdaptiveHistogramEqualizationImageFilter** | alpha | 0.3, 0.5, 1.0 | Más contraste en bordes (alpha alto) vs más suave (alpha bajo). |
| | beta | 0.3, 0.5, 1.0 | Más contraste en regiones planas (beta alto). |
| | radius | 3, 5 | Vecindad mayor → equalización más suave. |
| **MeanImageFilter** | radius | 2, 3, 4 | Mayor radio → más suavizado. |
| **MedianImageFilter** | radius | 2, 3, 4 | Mayor radio → más reducción de ruido. |
| **GradientMagnitudeImageFilter** | — | (ninguno) | Sin parámetros configurables en el script actual. |

---

## 4. Tabla resumen para experimentos

| Filtro | Parámetro | Valores sugeridos | Efecto esperado |
|--------|-----------|-------------------|-----------------|
| AdaptiveHistogramEqualizationImageFilter | alpha | 0.3, 0.5, 1.0 | Contraste en zonas de alto gradiente |
| AdaptiveHistogramEqualizationImageFilter | beta | 0.3, 0.5, 1.0 | Contraste en zonas uniformes |
| AdaptiveHistogramEqualizationImageFilter | radius | 3, 5 | Tamaño de vecindad para equalización local |
| MeanImageFilter | radius | 2, 3, 4 | Suavizado por promedio en vecindad |
| MedianImageFilter | radius | 2, 3, 4 | Suavizado y reducción de ruido |
| GradientMagnitudeImageFilter | — | — | Sin parámetros; resalta bordes |

---

## 5. Cómo ejecutar el runner

Desde la raíz del proyecto `class-filter-image`:

```bash
python run_all_filters.py
```

El script usa rutas relativas al directorio desde el que se ejecuta; no requiere rutas absolutas.

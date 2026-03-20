# Procesamiento de imágenes médicas — Filtros clásicos

**Autores:** Abel Albuez, Victoria Acero, Santiago Gil  
**Institución:** Pontificia Universidad Javeriana  
**Curso:** Procesamiento de imágenes médicas

## Resumen

Proyecto académico que implementa tres filtros de procesamiento de imágenes médicas sobre volúmenes 3D NIfTI usando ITK (Insight Toolkit). El trabajo se centra en métodos clásicos (no basados en aprendizaje profundo): filtro mediana, filtro Wiener adaptativo y mediana adaptativa, con un script runner que ejecuta todos los filtros sobre las imágenes de la carpeta `samples/` y genera paneles de comparación en PNG.

## Requisitos

- Python 3.x  
- pip  
- **Dependencias:** `itk>=5.4.5`, `numpy`, `matplotlib`, `jupyter`

## Configuración

**Crear y activar el entorno virtual**

```bash
python3 -m venv venv
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows (cmd):** `venv\Scripts\activate.bat`  
**Windows (PowerShell):** `venv\Scripts\Activate.ps1`

**Instalar dependencias:**

```bash
pip install -r requirements.txt
```

## Imágenes de prueba

Las imágenes deben descargarse desde **3D Slicer → Sample Data** y colocarse en la carpeta `samples/`. Se utilizan volúmenes NIfTI (`.nii` o `.nii.gz`), por ejemplo: `CTACardio.nii`, `MRBrainTumor1_3.nii`, `MRHead.nii.gz`, `USProstate_1.nii`.

## Uso

### Filtros individuales (CLI)

**Filtro mediana (ITK MedianImageFilter):**

```bash
python src/median.py <entrada.nii> <salida.nii> <radius>
```

**Filtro Wiener adaptativo:**

```bash
python src/wiener.py <entrada.nii> <salida.nii> --window M N
```

**Mediana adaptativa:**

```bash
python src/adaptive_median.py <entrada.nii> [salida.nii] [--max-window Smax]
```

### Ejecución en lote

```bash
python run_all_filters.py
```

Ejecuta los tres filtros sobre todas las imágenes en `samples/` mediante subproceso (CLI), guarda resultados en `result/` (subcarpetas por filtro) y genera figuras de comparación (corte axial: Original vs variantes filtradas) en `comparison_results/`.

## Filtros implementados

| Filtro | Script | Parámetros | Descripción |
|--------|--------|------------|-------------|
| **Mediana** | `src/median.py` | `radius` (int) | MedianImageFilter ITK. Mayor radio → más suavizado y reducción de ruido tipo sal y pimienta. |
| **Wiener adaptativo** | `src/wiener.py` | Ventana M×N | Filtro en dominio de la frecuencia; comportamiento según media y varianza local. |
| **Mediana adaptativa** | `src/adaptive_median.py` | Ventana inicial, Smax | Mediana con ventana dinámica; mejor que mediana fija ante ruido impulsivo denso y mejor preservación de bordes. |

## Salidas

- **`result/`** — Volúmenes filtrados en subcarpetas por filtro (`median_results/`, `wiener_results/`, `adaptive_median_results/`). Nombres del tipo `<base>_median_r3.nii`, `<base>_wiener.nii`, `<base>_adaptive_median.nii`.
- **`comparison_results/`** — PNG por filtro (p. ej. `median/`, `wiener/`, `adaptive_median/`) con figuras de comparación (Original | variantes) a 200 DPI para informes.

## Estructura del proyecto

```
taller-class-filter-image/
├── README.md
├── requirements.txt
├── run_all_filters.py
├── src/
│   ├── median.py
│   ├── wiener.py
│   └── adaptive_median.py
├── samples/           # Imágenes NIfTI (descargar desde 3D Slicer)
├── result/
│   ├── median_results/
│   ├── wiener_results/
│   └── adaptive_median_results/
├── comparison_results/
│   ├── median/
│   ├── wiener/
│   └── adaptive_median/
└── venv/
```

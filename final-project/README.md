# Proyecto Final — Segmentación clásica de gliomas (BraTS 2024 GLI Post-Treatment)

Segmentación **clásica (no deep learning)** de gliomas con **ITK / SimpleITK** sobre el
dataset *BraTS 2024 GLI Post-Treatment*. Curso de Procesamiento de Imágenes
Médicas — Pontificia Universidad Javeriana, 2026.

Autores: Abel Albuez, Victoria Acero, Santiago Gil

---

## Dataset

- **BraTS 2024 GLI Post-Treatment** — ~2,200 casos post-tratamiento, 7 instituciones.
- 4 modalidades MRI registradas a MNI (1 mm³ isotrópico, skull-stripped):
  `t1n`, `t1c`, `t2w`, `t2f`.
- Segmentación experta (`seg`) con 4 sub-regiones:
  - `1 = NETC` — Non-Enhancing Tumor Core
  - `2 = SNFH` — SubRegion FLAIR Hyperintensity (edema)
  - `3 = ET`   — Enhancing Tumor
  - `4 = RC`   — Resection Cavity (nuevo en BraTS 2024)

Los volúmenes `.nii.gz` **no se versionan** en el repo (ver `.gitignore`). Deben
descargarse de la página del challenge y colocarse en la ruta configurada por
`CASES_BASE` (notebook de EDA) o `IMAGES_DIR` (scripts de cada módulo).

---

## Pipeline

```
BraTS NIfTI  ──►  01_preprocesamiento  ──►  02_segmentacion  ──►  03_visualizacion
   (t1n,t1c,         denoising,                Otsu,                 overlays,
    t2w,t2f,         N4 bias field,            K-means / GMM,        mosaicos,
    seg)             normalización,            crecimiento de        métricas
                     registro                  regiones, watershed   (Dice / Jaccard)
```

- **EDA** (`notebooks/EDA_BraTS2024_GLI.ipynb`): justifica modalidades y métodos
  por sub-región a partir de histogramas, separabilidad y bimodalidad. Exporta un
  reporte HTML autocontenido en `notebooks/EDA_BraTS2024_GLI_reporte.html`.
- **01_preprocesamiento**: limpieza y armonización de intensidades; registro
  multimodal cuando aplique.
- **02_segmentacion**: implementación de los métodos clásicos por sub-región
  (Otsu por modalidad, clustering, crecimiento de regiones, watershed).
- **03_visualizacion**: overlays color-mapeados, mosaicos comparativos y métricas
  contra el ground truth (`seg`).

---

## Estructura del proyecto

```
final-project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── EDA_BraTS2024_GLI.ipynb            # EDA extendido (clásico)
│   ├── EDA_BraTS2024_GLI_reporte.html     # reporte autocontenido (entregable)
│   └── casos_demostrativos.csv            # subconjunto compartido por el equipo
├── Python/
│   ├── 01_preprocesamiento/
│   ├── 02_segmentacion/
│   └── 03_visualizacion/
├── images/                                # volúmenes NIfTI (no versionados)
└── output/                                # resultados de los módulos
```

---

## Instalación

Convención del curso: un `venv` llamado `.venv` dentro de `Python/`.

```bash
cd Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

---

## Cómo correr

1. **EDA y selección de casos demostrativos**

   ```bash
   jupyter lab notebooks/EDA_BraTS2024_GLI.ipynb
   ```

   Al final el notebook escribe:
   - `notebooks/EDA_BraTS2024_GLI_reporte.html` (entregable principal del EDA).
   - `notebooks/casos_demostrativos.csv` con 4–6 `case_id` que el resto del
     equipo debe reutilizar para que las comparaciones sean justas.

2. **Preprocesamiento** (`Python/01_preprocesamiento/`).
3. **Segmentación** (`Python/02_segmentacion/`).
4. **Visualización y métricas** (`Python/03_visualizacion/`).

Cada módulo trae su propio `README.md` con los parámetros y comandos.

---

## Convenciones del repositorio

- Comentarios y documentación en español.
- Rutas independientes del directorio de trabajo:

  ```python
  IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
  ```

- `matplotlib` siempre con `origin='lower'`.
- Mapa de colores compartido por todos los módulos:

  ```python
  LABEL_MAP = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
  ```

# Segmentación clásica del Tumor Realzante (ET) en BraTS 2024 GLI

Segmentación **automática del Tumor Realzante (ET, label 3)** sobre la modalidad **T1c**
con métodos **clásicos de SimpleITK**, **sin aprendizaje profundo**.

**Objetivo:** dado un caso de BraTS 2024 GLI, producir una máscara binaria del realce (ET)
sobre T1c usando umbralización, crecimiento de regiones, watershed y agrupamiento — y
evaluar sus límites frente al ground truth.

Curso de Procesamiento de Imágenes Médicas — Maestría en Ing. de Sistemas, Pontificia
Universidad Javeriana, 2026. Equipo: **Abel Albuez, Victoria Acero, Santiago Gil**.

---

## Dataset

- **BraTS 2024 GLI Post-Treatment.** Se lee **directamente desde los ZIP** en
  `final-project/datasets/` con `zipfile` — **no se descomprime** el archivo completo: cada
  caso/modalidad se extrae puntualmente a un temporal y se borra tras usarse.
- Modalidades por caso: `t1n`, `t1c`, `t2w`, `t2f` y la segmentación experta `seg`
  (1 mm³ isotrópico, skull-stripped). Etiquetas: `1=NETC`, `2=SNFH`, `3=ET`, `4=RC`.
- Subconjunto de trabajo: **`MAX_CASOS=100`** (configurable) del ZIP de *TrainingData*
  (el que tiene `seg`).
- `BraTS2024-...-ValidationData.zip` **no tiene etiquetas** → solo serviría para inferencia
  cualitativa, no para Dice.

| ZIP en `datasets/` | Etiquetas | Uso |
|---|---|---|
| `...-TrainingData.zip` | sí (`seg`) | **fuente principal** |
| `...-AdditionalTrainingData.zip` | sí | reserva |
| `...-ValidationData.zip` | **no** | solo inferencia cualitativa |

---

## Estructura del repositorio

```
final-project/
├── README.md
├── datasets/                      # ZIP de BraTS (no versionados)
├── scripts/
│   ├── requirements.txt
│   ├── .venv/                     # entorno (no versionado)
│   ├── comun/                     # paquete compartido
│   │   ├── constantes.py          # rutas (pathlib), etiquetas, sub-regiones, paleta
│   │   ├── io_zip.py              # lectura .nii.gz DIRECTA desde ZIP + E/S SimpleITK
│   │   ├── metricas.py            # Dice, Jaccard, sensibilidad, especificidad
│   │   └── reporte.py             # figuras .png + HTML autocontenido (base64)
│   ├── 01_eda/eda.py
│   ├── 02_limpieza/limpieza.py            (+ limpieza_core.py)
│   ├── 03_registro/registro.py
│   ├── 04_segmentacion/segmentacion.py    (+ seg_core.py)
│   └── 05_visualizacion/visualizacion.py  (+ viz_core.py)
└── output/                        # salidas por módulo (no versionadas)
    ├── eda/           outputs/  figuras/  eda_reporte.html
    ├── limpieza/      outputs/  figuras/  limpieza_reporte.html
    ├── registro/      outputs/  figuras/  registro_reporte.html
    ├── segmentacion/  outputs/  figuras/  segmentacion_reporte.html
    └── visualizacion/ figuras/            visualizacion_reporte.html
```

Cada módulo deja **tres tipos de salida**: volúmenes `.nii.gz`, figuras `.png` sueltas y un
**reporte HTML autocontenido con nombre propio** (`eda_reporte.html`, `limpieza_reporte.html`, …).

---

## Pipeline por módulos

| # | Módulo | Qué hace | Entradas | Salidas principales | Reporte |
|---|--------|----------|----------|---------------------|---------|
| 1 | **EDA** | Caracteriza geometría, intensidades, separabilidad ET vs sano y elige casos demostrativos | ZIP TrainingData (100 casos) | `parametros_eda.json`, `estadisticas_por_caso.csv`, `separabilidad.csv`, 5 PNG | `eda_reporte.html` |
| 2 | **Limpieza** | Solo T1c: **Wiener → N4 → normalización percentil [0.5, 99.5] → [0,1]** | T1c crudo (demostrativos) | `<caso>-t1c_limpio.nii.gz`, PNG antes/después | `limpieza_reporte.html` |
| 3 | **Registro** (demostrativo) | **Mattes MI + Euler3D rígido** multi-resolución: perturbación rígida conocida → recuperación | T1c (fijo) | `<caso>-desalineado/-recuperado.nii.gz`, `.tfm`, `metricas_registro.csv`, curva | `registro_reporte.html` |
| 4 | **Segmentación ET** | 4 métodos clásicos sobre T1c limpio: **Otsu, RegionGrowing, Watershed, GMM multimodal** | T1c limpio + `seg` (GT) | `<caso>-<metodo>-ET.nii.gz`, `metricas_segmentacion.csv`, `metricas_resumen.csv` | `segmentacion_reporte.html` |
| 5 | **Visualización** | Mosaicos 3 vistas (axial/sagital/coronal) centrados en el ET + análisis post-mortem | T1c limpio + máscaras del módulo 4 | mosaicos/overlays PNG | `visualizacion_reporte.html` |
| 6 | **Reconstrucción 3D** | Mallas ET predicho (RegionGrowing) vs GT + visor web | máscaras de segmentación + `seg` (GT) | `.obj`/`.glb` por caso + `metricas_3d.csv` | `reconstruccion3d_reporte.html` + `visor_3d.html` |

- El **registro** es **propio** del equipo (no proviene del código reutilizado de Santiago).
- La **segmentación** usa **9 casos**: 8 con ET (rango de volumen) + 1 sin ET (robustez).

---

## Resultados clave (números reales de las salidas)

**EDA** (`output/eda/outputs/parametros_eda.json`)
- 100 casos procesados, **78 con ET**.
- **T1c es la mejor modalidad para ET**: Fisher Discriminant Ratio mediano **1.32**
  (t2f 0.47, t2w 0.34, t1n 0.10).
- Umbral alto de Otsu en T1c (escala cruda): mediana **3120** (rango 352–6435) → la escala
  varía mucho entre casos, lo que justifica normalizar.
- Volumen de ET: mediana **4814 mm³** (rango 66–87 993).

**Registro** (`output/registro/outputs/metricas_registro.csv`)
- **TRE residual ≈ 0.007 mm** (sub-vóxel) → recupera la perturbación casi exactamente.
- Dice de cerebro **0.91 → 0.96**; MSE cae ~2 órdenes de magnitud.

**Segmentación de ET** (`output/segmentacion/outputs/metricas_resumen.csv`, Dice medio sobre los 8 casos con ET)

| Método | Multimodal | Dice | Jaccard | Sensibilidad | Especificidad |
|---|---|---|---|---|---|
| **RegionGrowing** | No | **0.320** | 0.196 | 0.362 | 0.999 |
| Watershed | No | 0.213 | 0.132 | 0.133 | 0.999 |
| Otsu | No | 0.164 | 0.097 | 0.516 | 0.995 |
| GMM multimodal | Sí | 0.094 | 0.052 | 0.630 | 0.965 |

**Hallazgo central:** los métodos clásicos por intensidad son **insuficientes** para ET —
ninguno supera **0.5 de Dice medio**. El **GMM multimodal sobre-segmenta** (la mayor
sensibilidad, 0.63, pero el peor Dice por exceso de falsos positivos). **Otsu falla** porque
el histograma de T1c es **unimodal** (coef. de bimodalidad < 5/9 en los casos analizados).

**Nota técnica (semilla):** en ET post-tratamiento el realce forma un anillo alrededor de la
**cavidad de resección (label 4, oscura)**, por lo que el **centroide** del ET cae en tejido
oscuro y arruinaba el crecimiento. Por eso el region growing se **siembra en el vóxel de ET
más brillante** y crece por **ventana brillante** `[α·I, max]` (α=0.65). El GT solo aporta la
ubicación de la semilla (inicialización semi-automática); la máscara final sale del
crecimiento sobre T1c, no del GT.

**Reconstrucción 3D** (`output/reconstruccion3d/outputs/metricas_3d.csv`)
- Mallas (marching cubes) de los 8 casos con ET: predicción (RegionGrowing) vs GT.
- El volumen de la malla GT reproduce el volumen de ET del EDA en el mismo orden de magnitud
  (p. ej. caso `02060-100`: malla GT **19 266 mm³** vs EDA **19 975 mm³**), lo que valida la
  escala física (spacing en mm).

---

## Cómo ejecutar

Requisitos: **Python 3.11**.

```bash
# 1) entorno (dentro de scripts/)
cd final-project/scripts
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) módulos en orden (cada uno deja .nii.gz + .png sueltos + su HTML en output/<modulo>/)
python 01_eda/eda.py --max_casos 100
python 02_limpieza/limpieza.py
python 03_registro/registro.py
python 04_segmentacion/segmentacion.py     # admite --no_product para el sweep posicional
python 05_visualizacion/visualizacion.py
python 06_reconstruccion3d/reconstruccion3d.py
```

Los módulos 2–6 leen `output/eda/outputs/parametros_eda.json` (casos demostrativos), así que
el **EDA debe ejecutarse primero**. El módulo 4 limpia automáticamente el T1c de los casos
nuevos que aún no estén en `output/limpieza/`. El módulo 6 reutiliza las máscaras del módulo 4.

**Visor 3D para la exposición.** El módulo 6 genera mallas en `.obj` (para Slicer/Blender) y
`.glb`, más un **visor web interactivo autocontenido** en `output/reconstruccion3d/visor_3d.html`
(Three.js servido desde `lib/` local — **funciona sin internet**): superpone el ET predicho
(naranja) y el GT (verde translúcido) por caso, con rotar/zoom/pan, toggles y el Dice del caso.
Dependencias extra del módulo 6: `trimesh` y `pygltflib` (ya en `requirements.txt`).

---

## Convenciones

- **SimpleITK** como motor de imagen; lectura de `.nii.gz` **directa desde los ZIP**.
- Rutas con **`pathlib`**, ancladas a la raíz del proyecto (nada hardcodeado).
- Comentarios en español; figuras con `origin='lower'`.
- Salidas organizadas bajo `output/<modulo>/` con reporte HTML de **nombre propio**.

---

## Crédito de reutilización

El núcleo numérico de varias etapas adapta **funciones puras** del paquete `brats_pipeline`
de **Santiago Gil** (auditoría en `AUDITORIA_BRATS_PIPELINE_SANTIAGO.md`): denoise (Adaptive
Wiener), N4, normalización, métricas (Dice/Jaccard, extendidas con sensibilidad/especificidad),
Otsu multinivel, crecimiento de regiones, watershed, GMM multimodal y los mosaicos de 3 vistas.
La capa de E/S se **reescribió** para leer desde los ZIP, y el **módulo de registro** (Mattes
MI + Euler3D) es **aporte propio del equipo**. El proyecto es un trabajo conjunto de los tres
autores.

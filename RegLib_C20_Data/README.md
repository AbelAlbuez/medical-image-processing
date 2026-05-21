# Taller 4 – Registro de Imágenes Médicas: RegLib Case #20 (Intra-subject whole-body PET-CT)

## 1. Descripción

Este taller aborda el **registro intra-sujeto** de un estudio PET-CT de cuerpo
completo correspondiente al caso #20 de la *Registration Library* del NA-MIC.
El objetivo es alinear dos adquisiciones de tomografía computarizada (CT_1 y
CT_2) del mismo paciente, obtenidas en momentos distintos, y posteriormente
**transferir la transformación estimada** sobre el par PET correspondiente,
aprovechando que cada PET se adquirió co-registrado con su CT.

El flujo de trabajo es:

1. Estimar la transformación geométrica que lleva `CT_1` → `CT_2` mediante un
   esquema de registro **Rigid + Affine** implementado en ITK.
2. Aplicar la transformación estimada a `PET_2` (que comparte sistema de
   coordenadas con `CT_2`) para llevarla al espacio de `PET_1`.
3. Visualizar y comparar los resultados en los tres planos anatómicos
   (axial, coronal, sagital) y calcular la imagen de diferencia PET.

## 2. Integrantes

- Abel Albuez Sanchez
- Victoria Acero
- Santiago Gil

Maestría en Ingeniería de Sistemas — Pontificia Universidad Javeriana, 2026.
Curso: *Procesamiento de Imágenes Médicas*.

## 3. Estructura del proyecto

```
RegLib_C20_Data/
├── images/                  # Volúmenes de entrada en formato .nrrd
│   ├── CT_1.nrrd            # CT de referencia (fija)
│   ├── CT_2.nrrd            # CT móvil (a registrar sobre CT_1)
│   ├── PET_1.nrrd           # PET asociado a CT_1
│   └── PET_2.nrrd           # PET asociado a CT_2
├── scripts/
│   ├── register_ct.py       # Registro CT_2 → CT_1 (Rigid + Affine, ITK)
│   ├── register_pet.py      # Aplica la transformación CT al PET_2
│   └── visualize_results.py # Genera vistas axial, coronal y sagital
├── results/                 # Imágenes .png generadas por los scripts
├── transforms/              # Transformaciones guardadas (.tfm / .h5)
└── README.md
```

## 4. Requisitos

- Python 3.9+
- `itk`
- `numpy`
- `matplotlib`
- `SimpleITK` (opcional, utilidades de E/S y visualización)

Instalación recomendada en un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install itk numpy matplotlib SimpleITK
```

## 5. Instrucciones de ejecución

Los scripts deben ejecutarse en el siguiente orden desde la raíz del módulo
`RegLib_C20_Data/`:

```bash
# Paso 1 — Registro CT_2 → CT_1 (Rigid + Affine).
#         Guarda la transformación final en transforms/.
python scripts/register_ct.py

# Paso 2 — Aplicación de la transformación CT al volumen PET_2.
#         Produce el PET registrado y lo deja disponible para visualización.
python scripts/register_pet.py

# Paso 3 — Visualización de resultados.
#         Guarda en results/ los cortes axial, coronal y sagital de
#         CT_1, CT_2 registrado, PET_2 registrado y la diferencia PET.
python scripts/visualize_results.py
```

Ninguno de los scripts abre ventanas interactivas: todas las figuras se
almacenan automáticamente en `results/` mediante `plt.savefig()`.

## 6. Nota sobre los datos

Las imágenes de entrada están en formato **NRRD** y deben ubicarse en la
carpeta `images/`. Debido a su tamaño no se incluyen en el repositorio Git;
es responsabilidad del usuario disponer de los volúmenes originales del caso
RegLib #20 antes de ejecutar los scripts.

## 7. Referencia

- *NA-MIC Registration Library — Case #20: Intra-subject whole-body PET-CT*.
  Disponible en: <https://www.na-mic.org/wiki/Projects:RegistrationLibrary:RegLib_C20>

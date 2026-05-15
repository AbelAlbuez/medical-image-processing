## Registro de Imágenes Médicas — Segunda Parte

### 1. Descripción
Este módulo implementa cuatro pipelines de registro de imágenes médicas usando ITK en Python, basados en los ejemplos oficiales del toolkit: ImageRegistration1, ImageRegistration2, ImageRegistration5 y ImageRegistration8.

### 2. Estructura del proyecto
```
registro-second-part/
├── Python/
│   ├── registro_4_pipelines.py
│   ├── ImageRegistration1.py
│   ├── ImageRegistration2.py
│   ├── ImageRegistration5.py
│   └── ImageRegistration8.py
├── imgs/
│   ├── BrainProtonDensitySliceBorder20.png
│   ├── BrainProtonDensitySliceR10X13Y17.png
│   ├── BrainProtonDensitySliceShifted13x17y.png
│   ├── BrainT1SliceBorder20.png
│   ├── brainweb1e1a10f20.mha
│   └── brainweb1e1a10f20Rot10Tx15.mha
├── output/
│   ├── pipeline1_traslacion2D_monomodal/
│   ├── pipeline2_traslacion2D_multimodal/
│   ├── pipeline3_rigido2D/
│   └── pipeline4_rigido3D/
├── requirements.txt
├── run.sh
└── README.md
```

### 3. Pipelines implementados
| Pipeline                          | Transformación           | Métrica           | Imágenes                                                     |
|-----------------------------------|--------------------------|-------------------|--------------------------------------------------------------|
| 1 — Traslación 2D Monomodal       | TranslationTransform     | MeanSquares       | BrainProtonDensitySliceBorder20 → SliceShifted13x17y        |
| 2 — Traslación 2D Multimodal      | TranslationTransform     | MutualInformation | BrainT1SliceBorder20 → SliceShifted13x17y                   |
| 3 — Rígido 2D Monomodal           | CenteredRigid2DTransform | MeanSquares       | BrainProtonDensitySliceBorder20 → SliceR10X13Y17            |
| 4 — Rígido 3D Monomodal           | VersorRigid3DTransform   | MeanSquares       | brainweb1e1a10f20 → brainweb1e1a10f20Rot10Tx15              |

### 4. Requisitos
- Python 3.8+
- Paquetes listados en requirements.txt: itk, numpy, matplotlib

### 5. Instalación y ejecución
Opción A — Script automático (recomendado):
```bash
cd registro-second-part
chmod +x run.sh
./run.sh
```

Opción B — Manual:
```bash
cd registro-second-part
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python Python/registro_4_pipelines.py
```

### 6. Salida esperada
Cada pipeline genera su propia carpeta dentro de output/ con 5 archivos:
- fixed.png — imagen fija original
- moving.png — imagen móvil original
- registered.png — imagen móvil después del registro
- difference.png — diferencia absoluta |fixed − registered|
- mosaico_<pipeline>.png — los 4 anteriores en una sola figura

### 7. Notas técnicas
- Todas las imágenes ITK se leen con pixel type itk.F.
- Pipeline 2 incluye la cadena de preprocesamiento: Cast → Normalize → DiscreteGaussianFilter (variance=2.0).
- Pipeline 4 visualiza el slice axial z=90 del volumen 3D.
- Los paths son absolutos relativos al script, por lo que funciona desde cualquier cwd.
- El venv y la carpeta output/ están en .gitignore.

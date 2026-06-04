"""
Paquete `comun`
===============
Utilidades compartidas por los 5 módulos del pipeline clásico de segmentación
del Tumor Realzante (ET, label 3) sobre T1c en BraTS 2024 GLI.

Submódulos:
  * constantes : rutas (pathlib), etiquetas BraTS, sub-regiones, paleta de reporte.
  * io_zip     : lectura de .nii.gz DIRECTAMENTE desde los ZIP + E/S SimpleITK.
  * metricas   : Dice, Jaccard, sensibilidad, especificidad y tablas comparativas.
  * reporte    : guardar figuras .png + embeber en base64 + armar HTML autocontenido.

Filosofía: estos módulos NO dependen de la capa de rutas/CSV de Santiago. Reutilizan
solo su núcleo numérico (denoise/N4/normalize/segmentación/métricas/figuras), que se
copia/adapta dentro de cada módulo correspondiente.
"""

# 01 — Preprocesamiento

Etapa de **limpieza y armonización** previa a la segmentación clásica.

## Objetivos

- **Denoising**: `CurvatureAnisotropicDiffusionImageFilter` o
  `GradientAnisotropicDiffusionImageFilter` (ITK) para reducir ruido sin
  difuminar bordes.
- **Corrección de bias field** N4 (`N4BiasFieldCorrectionImageFilter`) sobre cada
  modalidad antes de cualquier umbral o clustering.
- **Normalización de intensidades**: z-score por volumen (cerebro enmascarado) y
  *histogram matching* a una referencia común. Justificado por la sección D del
  EDA (variabilidad de rangos entre casos).
- **Registro multimodal** cuando aplique (verificación, ya viene MNI en BraTS).

## Convenciones

- Entrada: `final-project/images/<case_id>/<case_id>-<modalidad>.nii.gz`.
- Salida: `final-project/output/01_preprocesamiento/<case_id>/...`.
- Ruta independiente del CWD:

  ```python
  import os
  IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images")
  OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output", "01_preprocesamiento")
  ```

## Pendiente

Implementación pendiente. Tomar los `case_id` de
`notebooks/casos_demostrativos.csv` para mantener consistencia con los demás
módulos.

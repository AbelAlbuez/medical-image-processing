# 03 — Visualización y métricas

Genera **overlays**, **mosaicos comparativos** y reporta **métricas** de
segmentación contra el ground truth `seg` de BraTS.

## Salidas

- Overlays color-mapeados (mismo `COLOR_MAP` que el EDA: NETC rojo, SNFH verde,
  ET azul, RC amarillo).
- Mosaicos por caso × método × modalidad (estilo `taller-segmentacion-second`).
- Tabla de métricas por sub-región:
  - **Dice** (`2|A∩B| / (|A|+|B|)`)
  - **Jaccard** (`|A∩B| / |A∪B|`)
  - **Sensibilidad / Especificidad**
  - **Hausdorff 95** (opcional, `HausdorffDistanceImageFilter` de SimpleITK).

## Convenciones

- `matplotlib` siempre con `origin='lower'`.
- Entrada: `output/02_segmentacion/...` y `images/<case_id>/<case_id>-seg.nii.gz`.
- Salida: `output/03_visualizacion/...`.

## Pendiente

Implementación pendiente.

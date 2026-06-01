# 02 — Segmentación clásica por sub-región

Implementación de métodos **clásicos** (no deep learning) elegidos por
modalidad / sub-región según las conclusiones del EDA (secciones A–E).

## Métodos a implementar

| Método | Filtro ITK / SimpleITK | Sub-región sugerida | Modalidad principal |
|--------|------------------------|---------------------|---------------------|
| Otsu (multi-nivel) | `OtsuMultipleThresholdsImageFilter` | ET, SNFH | T1c, T2-FLAIR |
| K-means / GMM | `ScalarImageKmeansImageFilter` | NETC, ET | T1c |
| Crecimiento de regiones | `ConnectedThresholdImageFilter`, `ConfidenceConnectedImageFilter` | ET, RC | T1c, T1 |
| Watershed | `MorphologicalWatershedImageFilter` | TumorTotal | T2-FLAIR |

La asignación final (método ↔ sub-región ↔ modalidad) se ajusta con los
**coeficientes de solapamiento** y la **bimodalidad** reportados en las
secciones B y C del EDA.

## Convenciones

- Entrada: salidas de `01_preprocesamiento/` o `images/` directamente.
- Salida: `final-project/output/02_segmentacion/<metodo>/<case_id>/...`.
- Las semillas para crecimiento de regiones se inicializan con los **centroides
  e intensidades** reportadas por la sección E del EDA.

## Pendiente

Implementación pendiente. Usar los `case_id` de
`notebooks/casos_demostrativos.csv`.

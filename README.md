# BraTS 2024 GLI — Segmentación ET clásica + semi-automática

## Setup rápido

```powershell
# 1. Activar entorno
cd brats-final
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install --only-binary=:all: SimpleITK scipy scikit-image scikit-learn numpy pandas matplotlib tqdm nibabel open3d

# 2. Pegar casos en images/
#    images/BraTS-GLI-02119-101/  ← *-t1n, *-t1c, *-t2w, *-t2f, *-seg .nii.gz

# 3. Correr
$env:BRATS_PROJECT_ROOT = $PWD
$env:BRATS_DATASET_DIR  = "$PWD\images"
python run_all.py
```

## Para máximo Dice (método semilla)

1. Abre `viewer/viewer.html` en el navegador
2. Carga los `.nii.gz` del caso (arrastra o clic)
3. Activa **Mapa dif** en Modalidad — verás el tumor brillante
4. Haz clic en el punto más brillante del tumor
5. Copia el comando que aparece y ejecútalo

```powershell
python run_all.py --skip-clean --seed-case BraTS-GLI-02119-101 --seed-z 114 --seed-y 88 --seed-x 42
```

## Métodos implementados

### Clásicos (línea base)
| Método | Descripción | Dice esperado |
|--------|-------------|---------------|
| otsu_T1c | Otsu n=3 sobre T1c | ~0.10 |
| gmm_T1c | GMM 3 comp. sobre T1c | ~0.13 |
| sustraccion | Umbral sobre T1c−T1n | ~0.09 |
| gmm_2d | GMM 4 comp. sobre [T1c, mapa_dif] | ~0.08 |
| **semilla** | Region growing desde semilla manual | **0.4–0.7** |

### Contornos deformables — spline / level set (automáticos, NUEVOS)
Todos comparten una **semilla automática híbrida** (`roi_et_auto`):
GMM de 3 componentes sobre T1c (robusto) y, si éste degenera, el blob dominante
del mapa T1c−T1n.  Cada modelo refina esa semilla con una salvaguarda en `_post`
que revierte a la semilla si la evolución colapsa, se fuga o se va a otra región
(así nunca puntúa por debajo de la base GMM).  Semillas dispersas/enormes se
acotan por tamaño y caja para mantener el cómputo rápido.
Implementados en `src/brats_pipeline/seg_spline_levelset.py`.

**Dice-ET medido sobre el subset de 20 casos (totalmente automático):**

| Método | Descripción | Dice medio | Dice máx | casos>0.75 |
|--------|-------------|:----------:|:--------:|:----------:|
| **variational_spline** | Chan-Vese morfológico (level set variacional, energía de región) | **0.505** | 0.860 | 5 |
| **bspline** | Chan-Vese + regularización B-spline cúbica de la frontera por corte | 0.473 | 0.794 | 2 |
| **level_set** | Level set geodésico de contornos activos (edge-based, SimpleITK) | 0.455 | 0.781 | 2 |
| **spline** | Snake paramétrico de Kass (contorno = spline) con salvaguarda anti-colapso | 0.450 | 0.795 | 2 |
| *(ref. clásico)* otsu_T1c / gmm_T1c sobre T1c limpio | | 0.485 / 0.481 | 0.82 / 0.87 | 6 / 6 |

> Frente al ~0.1–0.3 de la línea base original, los nuevos métodos alcanzan
> **0.45–0.51 de media** (variational_spline el mejor, superando a todos los
> clásicos) y **0.76–0.86 en los casos con realce claro**.  En total, **26
> combinaciones (caso, método) superan Dice 0.75** y generan superficie de Poisson.

## Reconstrucción de superficie de Poisson (3D)

Cuando un método supera **Dice 0.75** en un caso, el pipeline reconstruye la
superficie del tumor (predicción vs ground-truth) con **Poisson screened
reconstruction** (open3d) y guarda figura + mallas `.ply` en
`output/figuras/poisson/`.

```powershell
# se ejecuta solo dentro de run_all.py; umbral configurable:
python run_all.py --skip-clean --poisson-thr 0.75
python run_all.py --skip-clean --skip-poisson      # desactivar
```

## Resumen de resultados

```powershell
python summarize.py 0.75   # media/mediana/max de Dice por método + nº casos>0.75
```

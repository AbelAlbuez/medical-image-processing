# crecRegiones-ITK

Proyecto de ejemplo para crecimiento de regiones usando ITK.

## Estructura

- `C++/`: implementaciones en C++ de `ConfidenceConnected` y `ConnectedThreshold`.
- `Python/`: scripts Python equivalentes para el mismo método de segmentación.
- `segmentacion_crecimiento_regiones.py`: script unificado que aplica ambos métodos sobre `A1_grayT1.nii.gz` y genera la figura comparativa.
- `setup_y_ejecutar.sh`: script bash que crea el entorno virtual, instala dependencias y ejecuta el script principal.
- `requirements.txt`: dependencias Python del proyecto.
- `A1_grayT1.nii.gz`: volumen de entrada (T1 cerebral).

## Dependencias (`requirements.txt`)

```
itk==5.4.0
itk-io==5.4.0
itk-filtering==5.4.0
itk-segmentation==5.4.0
numpy==1.26.4
matplotlib==3.8.4
```

## Setup: crear entorno virtual e instalar dependencias

Desde el directorio `crecRegiones-ITK/`:

1. Crear el entorno virtual (solo la primera vez):

   ```bash
   python3 -m venv venv
   ```

2. Activar el entorno virtual:

   ```bash
   source venv/bin/activate
   ```

3. Actualizar `pip` e instalar dependencias:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Ejecutar el script principal:

   ```bash
   python segmentacion_crecimiento_regiones.py
   ```

5. Salir del entorno virtual cuando termines:

   ```bash
   deactivate
   ```

### Alternativa: usar el script `setup_y_ejecutar.sh`

Ejecuta los cinco pasos anteriores de forma automática:

```bash
./setup_y_ejecutar.sh
```

## Archivos generados

Tras ejecutar `segmentacion_crecimiento_regiones.py` se crean en el mismo directorio:

- `segmentacion_manual.nii.gz`: salida del filtro `ConnectedThresholdImageFilter`.
- `segmentacion_automatica.nii.gz`: salida del filtro `ConfidenceConnectedImageFilter`.
- `comparacion_segmentacion.png`: figura 3×3 con cortes axiales y máscaras superpuestas.

## Descripción

Este subdirectorio contiene código y ejemplos para segmentación basada en crecimiento de regiones con ITK, usando tanto C++ como Python.

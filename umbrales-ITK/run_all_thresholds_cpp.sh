#!/usr/bin/env bash
# Ejecuta otsuThresholding, huangThresholding y triangleThresholding sobre
# todos los .nii / .nii.gz en samples/ y escribe resultados en output/.
#
# Requisito: compilar con CMake fuera del árbol fuente, por ejemplo:
#   cd umbrales-ITK/C++ && mkdir -p build && cd build
#   cmake .. && cmake --build .
#
# Uso:
#   ./run_all_thresholds_cpp.sh
#   SAMPLES_DIR=/ruta/samples OUTPUT_DIR=/ruta/out BIN_DIR=/ruta/build ./run_all_thresholds_cpp.sh
#   ./run_all_thresholds_cpp.sh 256   # número de bins (tercer argumento de cada filtro)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="${SAMPLES_DIR:-$SCRIPT_DIR/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
BIN_DIR="${BIN_DIR:-$SCRIPT_DIR/C++/build}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NBINS="$1"
else
  NBINS="${NBINS:-128}"
fi

OTSU="$BIN_DIR/otsuThresholding"
HUANG="$BIN_DIR/huangThresholding"
TRI="$BIN_DIR/triangleThresholding"

for exe in "$OTSU" "$HUANG" "$TRI"; do
  if [[ ! -f "$exe" ]] || [[ ! -x "$exe" ]]; then
    echo "No se encontró ejecutable: $exe" >&2
    echo "Compile el proyecto en C++ (cmake en un directorio build) y defina BIN_DIR si hace falta." >&2
    exit 1
  fi
done

if [[ ! -d "$SAMPLES_DIR" ]]; then
  echo "No existe la carpeta de muestras: $SAMPLES_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

empty=1
while IFS= read -r input_path; do
  [[ -z "$input_path" ]] && continue
  empty=0
  stem="$(basename "$input_path")"
  if [[ "$stem" == *.nii.gz ]]; then
    base="${stem%.nii.gz}"
  elif [[ "$stem" == *.nii ]]; then
    base="${stem%.nii}"
  else
    base="${stem%.*}"
  fi

  out_otsu="$OUTPUT_DIR/${base}_otsu.nii"
  out_huang="$OUTPUT_DIR/${base}_huang.nii"
  out_tri="$OUTPUT_DIR/${base}_triangle.nii"

  echo "Procesando: $stem (bins=$NBINS)"
  "$OTSU" "$input_path" "$out_otsu" "$NBINS"
  "$HUANG" "$input_path" "$out_huang" "$NBINS"
  "$TRI" "$input_path" "$out_tri" "$NBINS"
done < <(find "$SAMPLES_DIR" -maxdepth 1 -type f \( -name '*.nii' -o -name '*.nii.gz' \) | LC_ALL=C sort)

if [[ "$empty" -eq 1 ]]; then
  echo "No hay volúmenes (.nii / .nii.gz) en $SAMPLES_DIR" >&2
  exit 1
fi

echo "Listo. Salidas en: $OUTPUT_DIR"

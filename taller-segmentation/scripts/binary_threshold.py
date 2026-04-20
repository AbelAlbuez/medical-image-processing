#UMBRALES ENCONTRADOS MANUELAMENTE:
# Cerebro: 173 - 360
# Mama: 330 – 750 
# Higado: 44 - 85

import itk
from pathlib import Path
import sys
def get_integer_input(prompt, min_value=None, max_value=None): 
    while True:
        value_str = input(prompt).strip()
        try:
            value = int(value_str)
            if min_value is not None and value < min_value:
                print(f"El valor debe ser mayor o igual a {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"El valor debe ser menor o igual a {max_value}.")
                continue
            return value
        except ValueError:
            print("Ingrese un numero entero valido.")

def threshold_segmentation(input_path: Path, output_path: Path, lower: int, upper: int, outside_value: int = 0, inside_value: int = 255):
    InputPixelType = itk.F
    OutputPixelType = itk.UC
    Dimension = 3
    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(str(input_path))
    threshold_filter = itk.BinaryThresholdImageFilter[InputImageType, OutputImageType].New()
    threshold_filter.SetInput(reader.GetOutput())
    threshold_filter.SetLowerThreshold(lower)
    threshold_filter.SetUpperThreshold(upper)
    threshold_filter.SetOutsideValue(outside_value)
    threshold_filter.SetInsideValue(inside_value)
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(str(output_path))
    writer.SetInput(threshold_filter.GetOutput())
    writer.Update()


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "images"
    output_dir = base_dir / "threshold_results"
    if not input_dir.exists():
        print(f"No se encontró la carpeta: {input_dir}")
        sys.exit(1)
    image_files = sorted(list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz")))
    if not image_files:
        print(f"No se encontraron imagenes en: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(exist_ok=True)
    for i, img_path in enumerate(image_files, start=1):
        print("=" * 70)
        print(f"Imagen {i}/{len(image_files)}: {img_path.name}")
        print("Ingresa los valores de umbral para esta imagen.")
        lower = get_integer_input("  Umbral inferior: ")
        upper = get_integer_input("  Umbral superior: ")
        while upper < lower:
            print("El umbral superior no puede ser menor que el inferior.")
            upper = get_integer_input("  Umbral superior: ")
        output_name = img_path.name
        if output_name.endswith(".nii.gz"):
            output_name = output_name[:-7] + "_threshold.nii.gz"
        elif output_name.endswith(".nii"):
            output_name = output_name[:-4] + "_threshold.nii"
        output_path = output_dir / output_name
        try:
            threshold_segmentation(
                input_path=img_path,
                output_path=output_path,
                lower=lower,
                upper=upper,
                outside_value=0,
                inside_value=255,
            )
            print(f"Resultado guardado en: {output_path}\n")
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}\n")
    print("=" * 70)

if __name__ == "__main__":
    main()
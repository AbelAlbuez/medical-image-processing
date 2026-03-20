import itk
import numpy as np
image = itk.imread("Images/t1_icbm_normal_1mm_pn5_rf20.nii.gz", itk.F)
volume = itk.array_view_from_image(image).astype(np.float32)
output = np.copy(volume)
dim_z, dim_y, dim_x = volume.shape
for z in range(1, dim_z - 1):
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            neighborhood = volume[z-1:z+2, y-1:y+2, x-1:x+2]
            neighbors = neighborhood.flatten()
            neighbors_without_center = np.delete(neighbors, 13)
            output[z, y, x] = np.median(neighbors_without_center)
output_image = itk.image_view_from_array(output)
output_image.CopyInformation(image)
itk.imwrite(output_image, "resultado.nii.gz")

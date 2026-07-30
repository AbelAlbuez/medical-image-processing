# Surface-Based Evaluation and Reconstruction Analysis

Open3D Poisson reconstruction was run from ET boundary-voxel point clouds. Boundary clouds were capped at 8000 points and reconstructed at Poisson depth 7; surface metrics are symmetric point-set distances in millimeters: HD95, ASD, and squared-distance Chamfer.

## 1. GT Reconstruction Error Floor

| vol_bin | n  | floor_hd95_mean | floor_hd95_median | floor_asd_mean | floor_asd_median | floor_chamfer_median | fallback_count |
| ------- | -- | --------------- | ----------------- | -------------- | ---------------- | -------------------- | -------------- |
| small   | 31 | 6.9044          | 3.9575            | 2.5630         | 1.2605           | 1.7457               | 0              |
| medium  | 19 | 6.4654          | 6.5099            | 1.5983         | 1.4243           | 6.4128               | 0              |
| large   | 17 | 3.6815          | 2.5624            | 1.2404         | 1.1345           | 3.6931               | 0              |

Strongest geometry/error associations:

| floor_metric  | geometry_predictor     | n  | spearman_rho | p_value |
| ------------- | ---------------------- | -- | ------------ | ------- |
| floor_chamfer | gt_components          | 67 | 0.7375       | 0.0000  |
| floor_asd     | gt_components          | 67 | 0.7191       | 0.0000  |
| floor_hd95    | gt_components          | 67 | 0.6446       | 0.0000  |
| floor_chamfer | isoperimetric_quotient | 67 | -0.5165      | 0.0000  |
| floor_chamfer | thin_shell_proxy       | 67 | 0.5165       | 0.0000  |
| floor_asd     | thin_shell_proxy       | 67 | 0.3994       | 0.0008  |
| floor_asd     | isoperimetric_quotient | 67 | -0.3994      | 0.0008  |
| floor_chamfer | surface_area_proxy     | 67 | 0.3615       | 0.0026  |

## 2. Prediction Surface Fidelity

Large stratum, n=17:

| method             | vol_bin | n  | surface_hd95_mean | surface_hd95_median | surface_asd_mean | surface_asd_median | surface_chamfer_median | fallback_count | lesionwise_dice_mean | global_dice_mean |
| ------------------ | ------- | -- | ----------------- | ------------------- | ---------------- | ------------------ | ---------------------- | -------------- | -------------------- | ---------------- |
| variational_spline | large   | 17 | 52.4137           | 50.6212             | 18.4661          | 13.8552            | 469.0231               | 0              | 0.1762               | 0.3442           |
| gmm_T1c            | large   | 17 | 55.3921           | 49.9694             | 18.9220          | 14.1131            | 481.8570               | 0              | 0.1707               | 0.3291           |
| otsu_T1c           | large   | 17 | 51.2310           | 50.6288             | 19.6538          | 15.3971            | 481.2385               | 0              | 0.1650               | 0.3186           |
| sustraccion        | large   | 17 | 76.2375           | 75.6611             | 20.8895          | 20.7825            | 1123.7137              | 0              | 0.0087               | 0.3553           |
| gmm_2d             | large   | 17 | 76.5540           | 75.6875             | 21.7566          | 21.9960            | 1119.0143              | 0              | 0.0042               | 0.2226           |

All feasible strata:

| method             | vol_bin | n  | surface_hd95_mean | surface_hd95_median | surface_asd_mean | surface_asd_median | surface_chamfer_median | fallback_count | lesionwise_dice_mean | global_dice_mean |
| ------------------ | ------- | -- | ----------------- | ------------------- | ---------------- | ------------------ | ---------------------- | -------------- | -------------------- | ---------------- |
| variational_spline | large   | 17 | 52.4137           | 50.6212             | 18.4661          | 13.8552            | 469.0231               | 0              | 0.1762               | 0.3442           |
| gmm_T1c            | large   | 17 | 55.3921           | 49.9694             | 18.9220          | 14.1131            | 481.8570               | 0              | 0.1707               | 0.3291           |
| otsu_T1c           | large   | 17 | 51.2310           | 50.6288             | 19.6538          | 15.3971            | 481.2385               | 0              | 0.1650               | 0.3186           |
| sustraccion        | large   | 17 | 76.2375           | 75.6611             | 20.8895          | 20.7825            | 1123.7137              | 0              | 0.0087               | 0.3553           |
| gmm_2d             | large   | 17 | 76.5540           | 75.6875             | 21.7566          | 21.9960            | 1119.0143              | 0              | 0.0042               | 0.2226           |
| sustraccion        | medium  | 19 | 84.0949           | 86.9843             | 25.7010          | 26.0985            | 1589.0081              | 0              | 0.0054               | 0.0840           |
| gmm_2d             | medium  | 19 | 84.2279           | 84.8779             | 25.7764          | 26.2682            | 1577.4376              | 0              | 0.0028               | 0.0673           |
| gmm_T1c            | medium  | 19 | 74.7987           | 76.1911             | 29.8972          | 26.5358            | 1444.0204              | 0              | 0.0198               | 0.0283           |
| variational_spline | medium  | 19 | 75.1075           | 76.3990             | 29.5371          | 26.6023            | 1420.0189              | 0              | 0.0233               | 0.0330           |
| otsu_T1c           | medium  | 19 | 77.1158           | 78.2916             | 32.0617          | 29.2969            | 1520.7145              | 0              | 0.0031               | 0.0078           |
| sustraccion        | small   | 31 | 102.4684          | 101.6761            | 36.3140          | 34.9972            | 2521.2181              | 0              | 0.0034               | 0.0103           |
| gmm_2d             | small   | 31 | 102.0039          | 102.6211            | 36.3437          | 35.1195            | 2574.2201              | 0              | 0.0030               | 0.0095           |
| gmm_T1c            | small   | 31 | 102.4738          | 102.7984            | 51.7969          | 54.2482            | 3256.2881              | 0              | 0.0006               | 0.0016           |
| variational_spline | small   | 31 | 103.5320          | 103.8464            | 51.8875          | 54.3667            | 3293.7485              | 0              | 0.0007               | 0.0022           |
| otsu_T1c           | small   | 31 | 104.6814          | 107.7394            | 54.4576          | 56.9604            | 3365.1107              | 0              | 0.0002               | 0.0003           |

Dice-vs-surface relationship:

| vol_bin | n_methods | spearman_lesionwise_vs_surface_asd_median | p_asd  | spearman_lesionwise_vs_surface_hd95_median | p_hd95 |
| ------- | --------- | ----------------------------------------- | ------ | ------------------------------------------ | ------ |
| small   | 5         | -0.9000                                   | 0.0374 | -0.9000                                    | 0.0374 |
| medium  | 5         | 0.2000                                    | 0.7471 | -0.6000                                    | 0.2848 |
| large   | 5         | -1.0000                                   | 0.0000 | -0.9000                                    | 0.0374 |

## 3. Why Post-Treatment ET Is Geometrically Hard

Surface error by geometric situation:

| situation                   | n_case_method_rows | surface_asd_median | surface_hd95_median | surface_chamfer_median |
| --------------------------- | ------------------ | ------------------ | ------------------- | ---------------------- |
| unifocal                    | 85                 | 39.4791            | 102.6211            | 3071.2219              |
| multifocal                  | 235                | 27.0091            | 83.5931             | 1559.7771              |
| thin_shell_high_surface     | 170                | 26.2682            | 81.0425             | 1467.1484              |
| compact_low_surface         | 165                | 34.2561            | 93.6039             | 2421.8439              |
| low_isoperimetric_irregular | 170                | 26.2682            | 81.0425             | 1467.1484              |
| high_isoperimetric_compact  | 165                | 34.2561            | 93.6039             | 2421.8439              |

Irreducible old-20 confound cases:

| case_id             | method             | mechanism_class                                                         | gt_components | pred_recon_status | surface_hd95_vs_gt_recon | surface_asd_vs_gt_recon | surface_chamfer_vs_gt_recon | interpretation                                                      |
| ------------------- | ------------------ | ----------------------------------------------------------------------- | ------------- | ----------------- | ------------------------ | ----------------------- | --------------------------- | ------------------------------------------------------------------- |
| BraTS-GLI-00533-100 | gmm_T1c            | resection-margin / cavity-adjacent non-tumor enhancement                | 1             | ok                | 63.9526                  | 21.3402                 | 823.9543                    | cavity_or_vascular_confound_reconstructs_as_highly_separate_surface |
| BraTS-GLI-00533-100 | variational_spline | resection-margin / cavity-adjacent non-tumor enhancement                | 1             | ok                | 64.6209                  | 21.0836                 | 832.1684                    | cavity_or_vascular_confound_reconstructs_as_highly_separate_surface |
| BraTS-GLI-02078-100 | gmm_T1c            | midline vascular/dural-like non-tumor enhancement near resection region | 2             | ok                | 83.8182                  | 34.3892                 | 1881.9943                   | cavity_or_vascular_confound_reconstructs_as_highly_separate_surface |
| BraTS-GLI-02078-100 | variational_spline | midline vascular/dural-like non-tumor enhancement near resection region | 2             | ok                | 83.8620                  | 34.4936                 | 1884.7961                   | cavity_or_vascular_confound_reconstructs_as_highly_separate_surface |

Renders are saved in `phase2/surface_reconstruction/figures/`.

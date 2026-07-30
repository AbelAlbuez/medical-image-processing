# Next Steps

## Methodological Direction

The negative result points toward an integrated learned method rather than
another post-filter. The enhancement scalar is insufficient: tumor and
treatment-related enhancement can be equally bright, spatially plausible,
geometrically coherent, and highly persistent. A successful thesis method must
therefore make the discriminating information available before the mask is
formed.

## Failure-To-Method Map

| observed failure | implication | thesis method direction |
| --- | --- | --- |
| Enhancement/intensity cannot separate tumor from non-tumor enhancement. | A scalar threshold or GMM is underdetermined. | Use multimodal learned features from T1c, T1n, T2w, T2f and context around cavities/resection margins. |
| Location prior fails because hard confounds are not spatial outliers. | A post-hoc atlas veto is too blunt. | Learn spatial priors jointly with image evidence instead of masking after prediction. |
| Shape proxies separate components offline but fail as deployable post-filters. | Shape is useful, but the true/predicted component distribution shifts after segmentation. | Integrate shape/topology into the training objective or decoder constraints, not only as a rejection rule. |
| Cubical H0 persistence is inverted on enhancement maps. | Persistence on the scalar field inherits the confound. | If topology is used, compute it on learned probability fields or latent representations where tumor/confound separation has already been encouraged. |
| Surface ASD confirms voxel ranking. | Poor masks are also geometrically poor. | Optimize both overlap and boundary/surface-aware losses, especially for large tumors. |
| ET-absent cases flood across all classical methods. | The model needs a calibrated "no ET" decision. | Add absence-aware training, case-level detection heads, or uncertainty/calibration constraints. |

## Architecture Implication

The core architecture should be integrated rather than sequential. A plausible
next system is:

1. A multimodal encoder that sees the four BraTS modalities.
2. A segmentation decoder producing ET probability, not a hard intensity mask.
3. A spatial-context branch or learned atlas embedding.
4. A topology/shape-aware loss on the probability field, with care not to
encode a false spherical prior for post-treatment ET.
5. A case-level ET-presence head to reduce hallucination in ET-absent patients.
6. Boundary/surface-aware losses or validation metrics for the large-tumor
stratum.

The key lesson from Phase 2 is that topology must act on a representation where
the network has already learned some tumor/confound discrimination. Topology on
raw enhancement alone is inverted for the hard confounds.

## Paper-Completion Checklist

1. Add confidence intervals for all Stage-4 baseline summary tables:
   absent flood, median FP volume, large-stratum lesion-wise Dice, and surface
   ASD.
2. Run sensitivity excluding the three original-20 overlap cases:
   `02086`, `02143`, and `02151`.
3. Freeze any P2b operating point before new validation. Do not describe the
   current P2b point as confirmatory.
4. Verify bibliography:
   train-free cubical PH glioblastoma segmentation, SEDT/shape topology,
   Prastawa-style atlas priors, and SRI24/BraTS atlas references.
5. Audit terminology:
   R3 is shape-proxy geometry; R4 is persistent homology.
6. Add a raw-input determinism test or explicitly scope reproducibility to
   cleaned-input segmentation.
7. Add a methods paragraph disclosing threshold-sweep multiplicity and
   exploratory operating-point selection.
8. Keep tiny-n claims in the mechanism section only.
9. Prepare figure captions from `phase2/figures/CAPTIONS.md` for manuscript
   submission.
10. Freeze a clean release commit/tag after final tables and reports are
    accepted.

## Suggested Thesis Framing

The Phase-2 result should be framed as an impossibility result for classical
post-hoc correction:

> When tumor and treatment-related enhancement are collapsed into the same
> scalar evidence field, intensity, location, shape, and cubical H0 persistence
> each fail to separate them. The thesis method therefore learns an integrated
> multimodal representation in which topological and spatial constraints act on
> tumor-aware probabilities rather than on raw enhancement.

This converts the negative result into motivation: the failures are not dead
ends, they define the requirements for the learned system.


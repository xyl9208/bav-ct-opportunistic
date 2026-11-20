# Input data specification

## What the pipeline expects

The released code starts from **NIfTI volumes**, one file per examination, in a
directory per cohort:

```
$BAVCT_DATA/
├── internal/
│   ├── BAV/   <exam_id>.nii.gz
│   └── TAV/   <exam_id>.nii.gz
└── external/
    ├── anji/
    │   ├── BAV/    <exam_id>.nii.gz
    │   ├── TAV/    <exam_id>.nii.gz
    │   └── calib/  <exam_id>.nii.gz      # non-BAV calibration set
    └── quzhou/
        ├── BAV/
        ├── TAV/
        └── calib/
```

`<exam_id>` is any de-identified identifier; it is used as the join key between
the volume, the segmentation masks and the manifest.

## Volume requirements

- Non-gated, non-contrast chest CT, axial acquisition, full chest coverage
  (lung apices to costophrenic angles) so that the heart and ascending aorta are
  inside the field of view.
- Original DICOM geometry preserved in the NIfTI affine: millimetre spacing in
  the header, no prior resampling or cropping.
- Slice thickness is read from the header and used by the thickness
  randomisation and the per-thickness strata; do not rescale to isotropic
  spacing before running the pipeline.

## Segmentation masks

Masks are produced by `bavct.preprocess.segment`, which wraps
TotalSegmentator (`roi_subset=["aorta", "heart"]`) and writes, per examination,

```
$BAVCT_MASKS/<cohort>__<exam_id>/{aorta.nii.gz, heart.nii.gz}
```

Any external segmentation with the same two labels and the original grid can be
used instead; the measurement code only needs the two binary masks and the
voxel spacing.

## Manifest

`bavct.data.dataset` reads a CSV manifest at `$BAVCT_WORK/data/manifest.csv`
with these columns:

| column | meaning |
|---|---|
| `exam_no` | examination identifier, matches the NIfTI filename |
| `label` | `BAV` or `TAV` |
| `fold` | patient-level fold, 1 to 5 |
| `is_calib` | 1 if the examination belongs to the calibration set |
| `z_mm` | reconstructed slice thickness in millimetres |
| `patient_id` | de-identified patient key; folds are split on this column |

Column names are part of the interface. If the source tables use a
hospital-specific field name for the patient key, rename that column to
`patient_id` when exporting the manifest.

## Pre-processing manifest

`bavct.preprocess.roi` reads `$BAVCT_WORK/data/preprocess_manifest.csv`:

| column | meaning |
|---|---|
| `exam_no` | examination identifier |
| `label` | `BAV` or `TAV` |
| `ct` | path to the CT NIfTI volume |
| `aorta` | path to the aorta mask |
| `heart` | path to the heart mask |

Outputs are written to `$BAVCT_WORK/data/processed/c128/<label>/<exam_no>.npy`
and `.../v128/<label>/<exam_no>.npy` (128³, float16), with a quality-control
table at `$BAVCT_WORK/results/preprocess_qc.csv`. Cases below the central-block
mean or above the zero-voxel threshold are labelled `suspect_localization`.

## Cohort roster

`bavct.preprocess.segment` processes the examinations listed in
`$BAVCT_WORK/data/cohorts.csv`, one row per examination:

| column | meaning |
|---|---|
| `exam_no` | examination identifier |
| `cohort` | cohort key, e.g. `internal_BAV`, `anji_TAV` |

Which examinations enter each cohort is a study-level decision and is not part
of this repository; supply the roster for your own data.

## What is not distributed

No patient images, no derived arrays, no model weights and no score tables are
part of this repository. Full-precision regression parameters and covariance
matrices are available from the corresponding authors on reasonable request.

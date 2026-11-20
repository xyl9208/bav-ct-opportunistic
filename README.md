# Opportunistic detection of bicuspid aortic valve on routine noncontrast chest CT

Code accompanying the study *Opportunistic detection of bicuspid aortic valve on routine noncontrast chest CT using deep learning*.

A three-dimensional deep-learning (DL) classifier is trained on a 100-mm cubic region covering the aortic valve, aortic root and proximal ascending aorta, cropped from routine noncontrast chest CT acquired for noncardiac indications. Its score is combined with the automatically measured ascending aortic diameter, aortic root calcium volume, age and sex in a logistic regression — the joint model. This repository covers the analysis path from a CT series to the results reported in the paper.

## Contents

| Module | What it does |
|---|---|
| `preprocess` | cardiac and aortic segmentation (TotalSegmentator wrapper), valve-plane localisation, 100-mm composite and 56-mm valve cubes resampled to 128³, 3-mm z unification, slice-thickness randomisation; DICOM-level series selection is site-specific and not included |
| `measure` | Aortic centreline extraction, maximum chord diameter on planes perpendicular to the centreline with quality checks, and volumetric aortic root calcium by density threshold |
| `models` | Three-dimensional ResNet-18, five-fold patient-level training, ensemble inference, and the B-spline logistic regression of the clinical-imaging and joint models (`bavct.models.joint`) |
| `thresholds` | Comparison cutoffs and the fixed P95/P90 application cutoffs from independent negative calibration sets |
| `analysis` | Discrimination (ROC, AUC with DeLong intervals), Wilson intervals, paired bootstrap and exact McNemar comparisons, calibration and subgroup analyses; further study-level statistics are in `scripts` |
| `scripts` | Study drivers: aortic-root calcium batch, additional statistics, decision-curve analysis |
| `params` | Joint-model coefficients, spline knots and P95/P90 application cutoffs at the precision reported in the paper |
| `docs` | Input data specification, execution order and table provenance |

## Reproducibility boundary

Because the network weights are not released, the DL scores reported in the paper cannot be reproduced from this repository. Everything downstream of a DL score is fully specified and reproducible: the joint-model coefficients, variable transformations and spline knots are given at four decimals in Supplementary Table 4 and in `params/joint_model_coefficients.json`, and the fixed P95/P90 application cutoffs in Supplementary Table 5 and `params/thresholds.json`; full-precision parameters and covariance matrices are available from the corresponding authors on reasonable request. Given a DL probability, the joint-model score, its classification and the reported operating characteristics follow deterministically. No patient-level images, scores or trained weights are part of this repository.

## Requirements

Python 3.9 with PyTorch, MONAI, TotalSegmentator, SimpleITK, nibabel, pydicom, NumPy, SciPy, pandas, scikit-learn, statsmodels and matplotlib. Pinned versions correspond to Supplementary Table 3 of the paper and are listed in `environment/`.

All paths resolve through `bavct/config.py` and are overridden with two environment variables:

```bash
export BAVCT_WORK=/path/to/work    # manifests, runs, results
export BAVCT_DATA=/path/to/data    # cohort NIfTI tree
export PYTHONPATH=$PWD/src
```

The expected input layout and the order in which to run the modules are in `docs/input_spec.md` and `docs/pipeline.md`. The MedicalNet comparison arm additionally needs a local clone of the MedicalNet repository and its ResNet-18 checkpoint (`BAVCT_MEDICALNET_DIR`, `BAVCT_MEDICALNET_WEIGHTS`); neither is redistributed here and both are used under the terms of their authors. The deployed model does not use MedicalNet.

## Citation

Citation details will be added when the paper is published.

## Licence

MIT — see [LICENSE](LICENSE).

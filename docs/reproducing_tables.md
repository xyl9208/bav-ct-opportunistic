# Provenance of tables and figures

| Output | Produced by |
|---|---|
| Table 1 (cohort characteristics) | `scripts/additional_statistics.py` |
| Table 2 (discrimination, calibration) | `bavct.analysis.discrimination` |
| Table 3 / Supplementary Table 5 (thresholds) | `bavct.thresholds.calibrate` |
| Supplementary Table 4 (regression coefficients) | `params/joint_model_coefficients.json` |
| Calibration and decision-curve panels | `bavct.analysis.discrimination`, `scripts/decision_curve.py` |
| Subgroup forest plot data | `bavct.analysis.subgroups` |
| Pre-diagnostic detection | `scripts/additional_statistics.py` |
| Occlusion sensitivity | generated from the ensemble with `params/thresholds.json`; the block-wise occlusion loop is described in Methods |

The figure drawing scripts are not part of this release: they read patient-level
score tables, which cannot be distributed. The numerical outputs of the steps
above are sufficient to redraw every panel.

Coefficients and thresholds are reported at four decimals (Supplementary
Tables 4 and 5). Full-precision parameters and covariance matrices are available
from the corresponding authors on reasonable request.

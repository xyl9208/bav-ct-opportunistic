# Execution order

Set the two environment variables first:

```bash
export BAVCT_WORK=/path/to/work
export BAVCT_DATA=/path/to/data
export PYTHONPATH=$PWD/src
```

All modules resolve their paths through `bavct.config`; nothing is hard-coded.

| Step | Command | Output |
|---|---|---|
| 1. Segmentation | `python -m bavct.preprocess.segment <cohort> <shard> <nshard>` (reads `$BAVCT_WORK/data/cohorts.csv`) | `$BAVCT_MASKS/<cohort>__<exam>/` |
| 2. Valve-centred ROI | `python -m bavct.preprocess.roi --manifest work/data/preprocess_manifest.csv --workers 8` | `work/data/processed/{c128,v128}/<label>/`, `work/results/preprocess_qc.csv` |
| 3. Ascending-aorta diameter | library `bavct.measure.diameter` (no standalone CLI; called by the study driver) | diameter and quality-check tables |
| 4. Root calcification | `python scripts/root_calcium.py test` | `work/results/calc_root_v3_<set>.csv` |
| 5. Training (5 folds) | `python -m bavct.models.train --arm <arm> --fold <k>` | `work/runs/<arm>/f<k>/` |
| 6. Ensemble inference | `python -m bavct.models.infer` | out-of-fold and test scores |
| 7. Threshold calibration | `python -m bavct.thresholds.calibrate --scores work/results/calib_scores.csv --out work/results/site_thresholds.json --group-col site` | `work/results/site_thresholds.json` |
| 8. Statistics | `python -m bavct.analysis.discrimination` | AUC, sensitivity/specificity, calibration, DCA |
| 9. Subgroups | `python -m bavct.analysis.subgroups` | forest-plot data |
| 10. Additional analyses | `python scripts/additional_statistics.py`, `python scripts/decision_curve.py` | manuscript tables |

Cohort assembly, inclusion/exclusion and the construction of the two external
non-BAV calibration sets are not part of this repository; the code starts from
the cohort directories described in `docs/input_spec.md`.

## Notes

- The pipeline assumes non-gated, non-contrast chest CT and a single series per
  examination. DICOM-level series selection is site-specific (it depends on the
  local PACS and reporting system) and is not part of this repository; start
  from NIfTI volumes as described in `docs/input_spec.md`.
- Training uses five patient-level folds with a fixed seed; inference averages
  the five fold models by mean sigmoid.
- Steps 5–10 write only to `$BAVCT_WORK`; no step writes into the repository.

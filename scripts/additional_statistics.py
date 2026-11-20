#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Evaluate the author-approved frozen predictions; no model training/update.

Input: prediction_inputs.csv in this directory.
Authoritative Sp95 summaries: ../表格/performance_final.csv.
All outputs stay in this directory. Patient metadata are not audited.
"""
from bavct import config
from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint


OUT = Path(str(config.WORK)) / 'results'
OUT.mkdir(parents=True, exist_ok=True)
INPUT = OUT / 'prediction_inputs.csv'
FINAL = OUT / 'performance_final.csv'
MODELS = ('clinical', 'dl', 'joint')
COHORTS = ('internal', 'anji', 'quzhou', 'external_pooled')
PREVALENCES = (.005, .01, .02)
TARGET_SPECIFICITIES = (.95, .975, .99)
TARGET_SENSITIVITIES = (.85, .90)
LOGIT_EPSILON = 1e-6


def save_csv(rows, filename):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = OUT / filename
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return len(frame)


def workload(sensitivity, specificity, prevalence):
    tp = 1000 * prevalence * sensitivity
    fn = 1000 * prevalence * (1 - sensitivity)
    fp = 1000 * (1 - prevalence) * (1 - specificity)
    tn = 1000 * (1 - prevalence) * specificity
    alerts = tp + fp
    return {'prevalence': prevalence, 'detected_per1000': tp, 'missed_per1000': fn,
            'false_positive_per1000': fp, 'true_negative_per1000': tn,
            'alerts_per1000': alerts, 'PPV': tp / alerts if alerts else np.nan,
            'NPV': tn / (tn + fn) if tn + fn else np.nan,
            'NNR': alerts / tp if tp else np.nan}


def full_strict_roc(labels, prediction):
    """All distinct operating points for prediction > threshold; no thinning."""
    order = np.argsort(prediction, kind='mergesort')
    scores = prediction[order]
    outcomes = labels[order]
    distinct_last = np.flatnonzero(np.r_[np.diff(scores) != 0, True])
    thresholds = np.r_[np.nextafter(scores[0], -np.inf), scores[distinct_last]]
    removed_tp = np.r_[0, np.cumsum(outcomes)[distinct_last]]
    removed_fp = np.r_[0, np.cumsum(1 - outcomes)[distinct_last]]
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    tp = positives - removed_tp
    fp = negatives - removed_fp
    return pd.DataFrame({'threshold': thresholds, 'TP': tp.astype(int), 'FP': fp.astype(int),
                         'FN': removed_tp.astype(int), 'TN': removed_fp.astype(int),
                         'sensitivity': tp / positives, 'specificity': removed_fp / negatives})


def operating_point(labels, prediction, threshold):
    positive = prediction > threshold
    tp = int(np.sum(positive & (labels == 1)))
    fp = int(np.sum(positive & (labels == 0)))
    fn = int(np.sum(~positive & (labels == 1)))
    tn = int(np.sum(~positive & (labels == 0)))
    return {'threshold': float(threshold), 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


data = pd.read_csv(INPUT)
final = pd.read_csv(FINAL)
required = {'case_id', 'centre', 'label', *MODELS, 'diam_mm', 'calcium_mm3'}
assert required.issubset(data.columns), 'Missing required input column.'
assert data['label'].isin([0, 1]).all(), 'Outcome encoding must be 0/1.'
assert set(data['centre']) == {'internal', 'anji', 'quzhou'}, 'Unexpected cohort code.'
assert np.isfinite(data[list(MODELS)].to_numpy()).all(), 'Prediction must be finite.'
assert data[list(MODELS)].ge(0).all().all() and data[list(MODELS)].le(1).all().all()
final = final[final['model_id'].isin(MODELS)].set_index(['cohort_id', 'model_id'])

calibration_summary, calibration_bins = [], []
threshold_rows, workloads, dca_rows, checks, roc_frames = [], [], [], [], []

for cohort in COHORTS:
    frame = data[data['centre'].isin(['anji', 'quzhou'])] if cohort == 'external_pooled' else data[data['centre'].eq(cohort)]
    y = frame['label'].to_numpy(dtype=int)
    n, n_bav, n_tav = len(y), int(y.sum()), int((y == 0).sum())
    for model in MODELS:
        p = frame[model].to_numpy(dtype=float)
        clipped = np.clip(p, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
        lp = np.log(clipped / (1 - clipped))
        # Calibration evaluation only: do not replace p with fitted probabilities.
        citl_fit = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=lp).fit()
        slope_fit = sm.GLM(y, sm.add_constant(lp, has_constant='add'), family=sm.families.Binomial()).fit()
        citl_ci, slope_ci = np.asarray(citl_fit.conf_int()), np.asarray(slope_fit.conf_int())
        calibration_summary.append({'cohort': cohort, 'model': model, 'n': n, 'n_BAV': n_bav, 'n_TAV': n_tav,
                                    'observed_prevalence': n_bav / n,
                                    'CITL': float(citl_fit.params[0]), 'CITL_CI_low': citl_ci[0, 0], 'CITL_CI_high': citl_ci[0, 1],
                                    'calibration_intercept_free': float(slope_fit.params[0]),
                                    'calibration_slope': float(slope_fit.params[1]),
                                    'slope_CI_low': slope_ci[1, 0], 'slope_CI_high': slope_ci[1, 1],
                                    'Brier': float(np.mean((p - y) ** 2)),
                                    'evaluation_note': 'Original probabilities in the evaluation cohort; no probability update'})
        bins = pd.qcut(pd.Series(p), q=10, labels=False, duplicates='drop').to_numpy()
        for bin_id in np.unique(bins):
            keep = bins == bin_id
            bin_n, events = int(keep.sum()), int(y[keep].sum())
            ci_low, ci_high = proportion_confint(events, bin_n, alpha=.05, method='wilson')
            calibration_bins.append({'cohort': cohort, 'model': model, 'bin': int(bin_id + 1), 'n': bin_n,
                                     'n_BAV': events, 'mean_pred': float(p[keep].mean()), 'observed': events / bin_n,
                                     'observed_CI_low': ci_low, 'observed_CI_high': ci_high,
                                     'min_pred': float(p[keep].min()), 'max_pred': float(p[keep].max())})

        roc = full_strict_roc(y, p)
        assert (roc['TP'] + roc['FN']).eq(n_bav).all()
        assert (roc['FP'] + roc['TN']).eq(n_tav).all()
        assert roc['sensitivity'].is_monotonic_decreasing and roc['specificity'].is_monotonic_increasing
        selected = []
        for target in TARGET_SPECIFICITIES:
            threshold = np.quantile(p[y == 0], target, method='higher')
            result = operating_point(y, p, threshold)
            result.update(target_type='specificity', target=target)
            selected.append(result)
        for target in TARGET_SENSITIVITIES:
            # Highest threshold whose strict-positive sensitivity reaches the target.
            eligible = roc[roc['sensitivity'] >= target]
            threshold = float(eligible.iloc[-1]['threshold'])
            result = operating_point(y, p, threshold)
            result.update(target_type='sensitivity', target=target)
            selected.append(result)
        for result in selected:
            threshold_rows.append({'cohort': cohort, 'model': model, **result,
                                   **workload(result['sensitivity'], result['specificity'], .01)})

        reference = final.loc[(cohort, model)]
        sp95 = selected[0]
        auc = float(roc_auc_score(y, p))
        check = {'cohort': cohort, 'model': model, 'n': n,
                 'AUC_from_fixed_predictions': auc, 'authoritative_AUC': float(reference['AUC']),
                 'AUC_matches_reported_precision': abs(auc - float(reference['AUC'])) <= .00005001,
                 'Se_at_Sp95_from_fixed_predictions': sp95['sensitivity'],
                 'authoritative_Se_at_Sp95': float(reference['Se_at_Sp95']),
                 'Se_matches_reported_precision': abs(sp95['sensitivity'] - float(reference['Se_at_Sp95'])) <= .00050001,
                 'n_matches_final_table': n_bav == int(reference['n_BAV']) and n_tav == int(reference['n_TAV'])}
        checks.append(check)
        if not (check['AUC_matches_reported_precision'] and check['Se_matches_reported_precision'] and check['n_matches_final_table']):
            raise AssertionError('Frozen-prediction metrics do not match the author-final table: ' + cohort + '/' + model)

        approved_se = float(reference['Se_at_Sp95'])
        approved_sp = .95
        for prevalence in PREVALENCES:
            workloads.append({'cohort': cohort, 'model': model, 'sensitivity': approved_se, 'specificity': approved_sp,
                              **workload(approved_se, approved_sp, prevalence),
                              'estimate_type': 'Prevalence-standardized scenario based on author-final Se@Sp95 and nominal Sp=.95'})
            for pt in np.linspace(.001, .05, 491):
                nb = prevalence * approved_se - (1 - prevalence) * (1 - approved_sp) * pt / (1 - pt)
                dca_rows.append({'cohort': cohort, 'model': model, 'prevalence': prevalence,
                                 'threshold_probability': pt, 'sensitivity': approved_se, 'specificity': approved_sp,
                                 'net_benefit': nb, 'analysis': 'Fixed Sp95 referral strategy; preference varies, predictions do not'})

        roc.insert(0, 'model', model)
        roc.insert(0, 'cohort', cohort)
        roc['prevalence'] = .01
        roc['detected_per1000'] = 1000 * .01 * roc['sensitivity']
        roc['false_positive_per1000'] = 1000 * .99 * (1 - roc['specificity'])
        roc['alerts_per1000'] = roc['detected_per1000'] + roc['false_positive_per1000']
        roc_frames.append(roc)

    # Default reference strategies, exported once per cohort/prevalence.
    for prevalence in PREVALENCES:
        for pt in np.linspace(.001, .05, 491):
            for model, se, sp in [('refer_all', 1., 0.), ('refer_none', 0., 1.)]:
                nb = prevalence * se - (1 - prevalence) * (1 - sp) * pt / (1 - pt)
                dca_rows.append({'cohort': cohort, 'model': model, 'prevalence': prevalence,
                                 'threshold_probability': pt, 'sensitivity': se, 'specificity': sp,
                                 'net_benefit': nb, 'analysis': 'Default referral strategy'})

counts = {}
counts['calibration_summary.csv'] = save_csv(calibration_summary, 'calibration_summary.csv')
counts['calibration_bins.csv'] = save_csv(calibration_bins, 'calibration_bins.csv')
counts['threshold_operating_points.csv'] = save_csv(threshold_rows, 'threshold_operating_points.csv')
counts['workload_scenarios.csv'] = save_csv(workloads, 'workload_scenarios.csv')
counts['strategy_dca.csv'] = save_csv(dca_rows, 'strategy_dca.csv')
counts['roc_workload_full.csv'] = save_csv(pd.concat(roc_frames, ignore_index=True), 'roc_workload_full.csv')
counts['consistency_checks.csv'] = save_csv(checks, 'consistency_checks.csv')

contract = {'models': list(MODELS), 'cohorts': list(COHORTS), 'probability_units': '0 to 1',
            'threshold_rule': 'prediction > threshold',
            'specificity_target_rule': 'TAV-score quantile, method=higher',
            'sensitivity_target_rule': 'largest distinct threshold attaining sensitivity >= target',
            'calibration': {'CITL': 'Binomial GLM intercept; offset=logit(original p), slope fixed at 1',
                            'slope': 'Binomial GLM with free intercept and logit(original p)',
                            'intervals': 'GLM Wald 95% CI; Wilson 95% CI for bin observations',
                            'bins': '10 quantile bins; tied probabilities remain together, duplicate cuts removed if necessary',
                            'logit_clip': [LOGIT_EPSILON, 1 - LOGIT_EPSILON],
                            'Brier': 'Mean squared error using unchanged original probabilities',
                            'probability_update': False},
            'workload_source': 'Author-final Se_at_Sp95 with nominal specificity .95',
            'DCA_type': 'Fixed binary strategy; not continuous-risk DCA',
            'DCA_prevalences': list(PREVALENCES), 'DCA_pt_min': .001, 'DCA_pt_max': .05, 'DCA_pt_step': .0001,
            'DCA_net_benefit_units': 'per person; multiply by 1000 for per-1000 display',
            'ROC_workload_prevalence': .01, 'ROC_thinning': False,
            'metadata_reaudit': False, 'model_retraining': False,
            'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            'authoritative_table_sha256': hashlib.sha256(FINAL.read_bytes()).hexdigest(),
            'software_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']},
            'outputs_and_rows': counts,
            'calibration_bins_per_model': pd.DataFrame(calibration_bins).groupby(['cohort', 'model']).size().rename('bins').reset_index().to_dict('records')}
(OUT / 'analysis_outputs_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(counts, ensure_ascii=False))
print('All fixed-prediction AUC, Sp95 sensitivity and cohort-size checks agree with the author-final table.')

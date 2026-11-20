#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Deterministic DCA from the author-approved fixed-threshold summary table.

No patient data, model fitting, AUC calculation, or probability recalibration.
"""
from pathlib import Path

from bavct import config
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent

SOURCE = Path(str(config.WORK)) / 'clinical_models'
SITES = [('内部时间外', 'internal', 'Internal temporal'),
         ('安吉', 'anji', 'External centre 1'),
         ('衢州', 'quzhou', 'External centre 2')]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build():
    with (SOURCE / 'test_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        approved = list(csv.DictReader(stream))
    cutoffs = json.loads((SOURCE / 'dev_thresholds.json').read_text())['THR']
    ops = []
    for source_label, key, display in SITES:
        source_row, = [row for row in approved if row['set'] == source_label and row['model'] == '联合模型']
        for i, point in enumerate(['P95', 'P90']):
            suffix = point[1:]
            ops.append({'cohort': key, 'cohort_label': display, 'model': 'joint',
                        'model_label': 'Combined model', 'operating_point': point,
                        'score_cutoff': cutoffs[f'联合模型|{key}'][i],
                        'sensitivity': float(source_row[f'Se{suffix}']) / 100,
                        'specificity': float(source_row[f'Sp{suffix}']) / 100,
                        'sensitivity_percent': float(source_row[f'Se{suffix}']),
                        'specificity_percent': float(source_row[f'Sp{suffix}'])})
    write_csv(HERE / 'joint_fixed_policy_operating_points.csv', ops)
    metadata = {'model': 'Combined model', 'author_approved_source': 'Fixed-threshold test summary and development thresholds',
                'source_sha256': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                  for name in ['test_results.csv', 'dev_thresholds.json']},
                'source_units': 'Se95/Sp95/Se90/Sp90 are percentages; divided by 100 for calculations',
                'operating_points': ops, 'target_prevalences': [0.005, 0.01, 0.02],
                'threshold_probability_range': [0.001, 0.25], 'threshold_probability_step': 0.0001,
                'formula': 'NB = prevalence*sensitivity - (1-prevalence)*(1-specificity)*pt/(1-pt)',
                'refer_all_formula': 'NB = prevalence - (1-prevalence)*pt/(1-pt)',
                'refer_none_formula': 'NB = 0',
                'no_new_fit': True, 'no_AUC_calculation': True,
                'no_patient_level_data_access': True, 'no_probability_update': True,
                'uncertainty': 'No confidence bands; deterministic scenario calculations from supplied summary estimates'}
    (HERE / 'joint_fixed_policy_operating_points.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    rows = []
    for prevalence in metadata['target_prevalences']:
        for step in range(10, 2501):
            pt = step / 10000
            weight = pt / (1 - pt)
            for op in ops:
                nb = prevalence * op['sensitivity'] - (1 - prevalence) * (1 - op['specificity']) * weight
                rows.append({'prevalence': prevalence, 'cohort': op['cohort'], 'cohort_label': op['cohort_label'],
                             'model': 'joint', 'operating_point': op['operating_point'],
                             'score_cutoff': op['score_cutoff'], 'sensitivity': op['sensitivity'],
                             'specificity': op['specificity'], 'threshold_probability': pt,
                             'net_benefit': nb, 'net_benefit_per1000': 1000 * nb})
            for policy, nb in [('refer_all', prevalence - (1 - prevalence) * weight), ('refer_none', 0.0)]:
                rows.append({'prevalence': prevalence, 'cohort': 'all', 'cohort_label': 'All centres',
                             'model': policy, 'operating_point': '', 'score_cutoff': '',
                             'sensitivity': 1 if policy == 'refer_all' else 0,
                             'specificity': 0 if policy == 'refer_all' else 1,
                             'threshold_probability': pt, 'net_benefit': nb, 'net_benefit_per1000': 1000 * nb})
    write_csv(HERE / 'joint_fixed_policy_dca.csv', rows)
    print(json.dumps({'operating_points': len(ops), 'curve_rows': len(rows), 'decision_thresholds_per_curve': 2491,
                      'prevalences': metadata['target_prevalences']}, ensure_ascii=False))


if __name__ == '__main__':
    build()

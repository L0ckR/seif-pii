#!/usr/bin/env python3
"""Compare numeric/hash-only replay outputs; no corpus or inference needed."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    path = ROOT / name
    raw = path.read_bytes() if path.exists() else gzip.decompress((ROOT / (name + '.gz')).read_bytes())
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


baseline, baseline_hash = read('baseline.json')
current, current_hash = read('current.json')
reference, _ = read('reference.json')
reference_checks = {key: digest(baseline[key]) == value for key, value in reference['canonical_field_sha256'].items()}
fields = ('raw_predictions', 'masked_text_sha256', 'actual_masked_alnum_offsets',
          'paired_mapped_spans', 'gold_alnum_offsets', 'scopes', 'input_sha256', 'mapping', 'excluded_rows')
checks = {field: baseline[field] == current[field] for field in fields}
modes = ('rules', 'person_only', 'person_location')
per_mode = {}
for mode in modes:
    ids = sorted(baseline['raw_predictions'][mode])
    assert ids == sorted(current['raw_predictions'][mode])
    changed = {field: [key for key in ids if baseline[field][mode][key] != current[field][mode][key]]
               for field in ('raw_predictions', 'masked_text_sha256', 'actual_masked_alnum_offsets', 'paired_mapped_spans')}
    per_mode[mode] = {'cases': len(ids), 'changed_case_ids': changed,
                     'changed_case_counts': {field: len(values) for field, values in changed.items()},
                     'before_raw_prediction_count': sum(len(x) for x in baseline['raw_predictions'][mode].values()),
                     'after_raw_prediction_count': sum(len(x) for x in current['raw_predictions'][mode].values()),
                     'case_ids_sha256': digest(ids)}
restored = all(x['validation']['actual_mask_and_restore_checked_all_modes_all_rows'] for x in (baseline, current))
passed = all(checks.values()) and all(reference_checks.values()) and restored
report = {'schema_version': 1, 'status': 'PASS' if passed else 'FAIL',
          'baseline_revision': baseline['revision'], 'current_revision': current['revision'],
          'offered_rows': baseline['offered_rows'], 'aligned_rows': baseline['aligned_rows'],
          'profile_row_pairs': sum(x['cases'] for x in per_mode.values()),
          'excluded_rows': baseline['excluded_rows'], 'checks': checks,
          'baseline_reproduces_accepted_121': reference_checks,
          'actual_mask_restore_all_rows_both_versions': restored, 'modes': per_mode,
          'input_sha256': baseline['input_sha256'],
          'source_sha256': {'baseline': baseline['source_sha256'], 'current': current['source_sha256']},
          'runner_sha256': baseline['runner_sha256'],
          'runner_identical_between_versions': baseline['runner_sha256'] == current['runner_sha256'],
          'comparison_scope': ['Ordered raw detector/merged type,start,end triples, before mapping or adjacent-span merging.',
              'SHA-256 of full actual masked string, all output types, plus exact per-position alphanumeric mask offsets.',
              'Actual mask→restore_exact round trip for every row/profile/version.',
              'All frozen typed-character and exact-span metric scopes, labels, mappings and exclusions.'],
          'canonical_output_sha256': {side: {field: digest(value[field]) for field in fields}
                                     for side, value in [('baseline', baseline), ('current', current)]},
          'metrics': {name: {mode: {metric: {'before': baseline['scopes'][name]['systems'][mode][metric],
                                           'after': current['scopes'][name]['systems'][mode][metric]}
                                  for metric in ('typed_character', 'merged_exact_span')}
                            for mode in modes} for name in ('common5', 'all_mapped8')},
          'runtime_python': {'baseline': baseline['python'], 'current': current['python']},
          'limitations': ['Offline replay of previously used development corpus, not a new quality estimate.',
                          'Frozen PERSON/LOCATION offsets use the established reconstructed score 0.85.',
                          'Prediction equality concerns type/start/end/order; confidence/reason fields are not compared.',
                          'Equal masking is evaluated in mask mode; no randomized token comparison is performed.',
                          'Existing quality limitations of the accepted baseline remain unchanged.'],
          'bootstrap': 'Not performed; all compared outputs are identical.' if passed else 'Not performed; inspect reported differences.'}
(ROOT / 'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
summary = [f"# RedMadRobot code-quality parity: {report['status']}", '',
    f"`{baseline['revision']}` → `{current['revision']}`.", '',
    f"Проверено {report['aligned_rows']} строк × {len(modes)} режима = {report['profile_row_pairs']} пар. Два прежних исключения сохранены.", '',
    'Сравниваются полные упорядоченные type/start/end списки до объединения типов, SHA-256 полного маскированного текста, все позиции замаскированных букв/цифр, восстановление исходного текста и все замороженные срезы метрик.', '',
    '| Режим | Строк | Изменённые списки сущностей | Изменённые маски |', '|---|---:|---:|---:|']
for mode, value in per_mode.items():
    summary.append(f"| {mode} | {value['cases']} | {value['changed_case_counts']['raw_predictions']} | {value['changed_case_counts']['masked_text_sha256']} |")
summary += ['', 'Baseline точно воспроизводит ранее принятую версию121 по всем сохранённым метрикам, offsets, маскам и протоколу.' if all(reference_checks.values()) else 'Baseline reference validation failed.',
    'Все исходные ошибки/ограничения качества принятой версии сохранены; это проверка отсутствия изменения поведения после рефакторинга.', '']
(ROOT / 'summary.md').write_text('\n'.join(summary))
validation = {'status': report['status'], 'network_disabled_both_versions': all(x['validation']['network_disabled'] for x in (baseline,current)),
              'source_and_inputs_immutable_both_versions': all(x['validation']['source_and_inputs_immutable'] for x in (baseline,current)),
              'no_model_inference': all(not x['validation']['live_ner_inference'] for x in (baseline,current)),
              'roundtrip_checks': 2 * report['profile_row_pairs'],
              'artifact_sha256': {'baseline.json': baseline_hash, 'current.json': current_hash,
                  **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                     ('runner.py','aggregate.py','reference.json','report.json','summary.md')}}}
(ROOT / 'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'status':report['status'],'profile_row_pairs':report['profile_row_pairs'],
                  'changes':{mode: value['changed_case_counts'] for mode,value in per_mode.items()},
                  'baseline_reference_identical':all(reference_checks.values())}))

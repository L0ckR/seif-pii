#!/usr/bin/env python3
"""Standalone numeric/hash-only Scanpatch parity aggregation."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    p = ROOT/name
    raw = p.read_bytes() if p.exists() else gzip.decompress((ROOT/(name+'.gz')).read_bytes())
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


baseline, baseline_sha = read('baseline.json')
current, current_sha = read('current.json')
reference, _ = read('reference.json')
fields = ('raw_predictions','masked_text_sha256','actual_masked_alnum_offsets','mapped_predictions',
          'fine_gold','mapped_gold','mapping','scopes','input_sha256','support_source_sha256','score_reconstruction_evidence')
checks = {k:baseline[k] == current[k] for k in fields}
modes = ('rules','hybrid')
reference_checks = {mode:digest({k:sorted(v)for k,v in baseline['raw_predictions'][mode].items()}) == reference['sorted_raw_prediction_sha256'][mode] for mode in modes}
reference_checks['scopes'] = digest(baseline['scopes']) == reference['scopes_sha256']
changes = {mode:{field:[k for k,v in baseline[field][mode].items()if current[field][mode][k] != v]
                for field in ('raw_predictions','masked_text_sha256','actual_masked_alnum_offsets','mapped_predictions')}
           for mode in modes}
restored = all(x['validation']['mask_restore_all_profile_rows_passed']for x in (baseline,current))
passed = all(checks.values()) and all(reference_checks.values()) and restored
report = {'schema_version':1,'status':'PASS' if passed else 'FAIL',
          'baseline_revision':baseline['revision'],'current_revision':current['revision'],
          'cases':baseline['cases'],'primary_cases':baseline['primary_cases'],'profiles':list(modes),
          'profile_row_pairs':baseline['cases']*len(modes),'roundtrip_checks_both_versions':baseline['cases']*len(modes)*2,
          'checks':checks,'baseline_reproduces_accepted_121':reference_checks,
          'actual_mask_restore_all_rows_both_versions':restored,'changed_case_ids':changes,
          'changed_case_counts':{mode:{field:len(values)for field,values in rows.items()}for mode,rows in changes.items()},
          'input_sha256':baseline['input_sha256'],
          'source_sha256':{'baseline':baseline['source_sha256'],'current':current['source_sha256']},
          'support_source_sha256':baseline['support_source_sha256'],
          'runner_sha256':baseline['runner_sha256'],'runner_identical_between_versions':baseline['runner_sha256']==current['runner_sha256'],
          'python':{'baseline':baseline['python'],'current':current['python']},
          'canonical_output_sha256':{side:{field:digest(value[field])for field in fields}for side,value in [('baseline',baseline),('current',current)]},
          'scopes':baseline['scopes'],
          'limitations':['Offline behavior parity on already-used frozen corpus; not a new quality estimate.',
                         'Original 532 rows, labels, coarse mapping and 164 primary whole cases retained; no per-language inference.',
                         'Fixed accepted evaluator helpers are used for both detectors; actual detector/transform source uses the selected Git revision.',
                         'Only type/start/end/order are compared, not confidence or reason fields.',
                         'Actual masking and exact restoration are checked in mask mode; randomized token strings are not compared.',
                         'Original PERSON/LOCATION offsets are reused at validated score 0.85; no model inference or bootstrap.']}
(ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=[f"# Scanpatch behavior parity: {report['status']}",'',f"`{baseline['revision']}` → `{current['revision']}`.",'',
       f"Проверено {report['cases']} строк × {len(modes)} режима = {report['profile_row_pairs']} пар; {report['roundtrip_checks_both_versions']} реальных mask→restore проверок.",'',
       '| Режим | Изменённые raw type/start/end списки | Изменённые маски |','|---|---:|---:|']
for mode in modes:
    lines.append(f"| {mode} | {len(changes[mode]['raw_predictions'])} | {len(changes[mode]['masked_text_sha256'])} |")
lines+=['','Сравнены полные упорядоченные spans до объединения типов, SHA-256 полных масок, точные позиции закрытых букв/цифр, mapped spans и все пять прежних срезов метрик.','',
        'Baseline воспроизводит принятые121 предсказания и метрики. Прежние ограничения качества сохранены. Исходные тексты и разметка не менялись; публикация содержит числа/offsets/hashes без текстов.','']
(ROOT/'summary.md').write_text('\n'.join(lines))
validation={'status':report['status'],'network_disabled_both_versions':all(x['validation']['network_disabled']for x in (baseline,current)),
            'no_live_inference':all(not x['validation']['live_model_inference']for x in (baseline,current)),
            'source_and_inputs_immutable':all(x['validation']['sources_and_inputs_immutable']for x in (baseline,current)),
            'actual_roundtrip_checks':report['roundtrip_checks_both_versions'],
            'artifact_sha256':{'baseline.json':baseline_sha,'current.json':current_sha,**{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('runner.py','aggregate.py','reference.json','report.json','summary.md')}}}
(ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'status':report['status'],'pairs':report['profile_row_pairs'],'changed':report['changed_case_counts'],'accepted_baseline_reference':reference_checks}))

"""Aggregate paired-case and stratum audit, using frozen offsets and domain metadata."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TASK = Path('/home/lockr/projects/seif-pii-golden-improvements')
CACHE = Path('/home/lockr/projects/seif-pii/output/external-bench')
PRIOR = TASK / 'output/golden-improvements/external-pii-bench'
protocol = json.loads((CACHE / 'prepared-protocol.json').read_text())
mapping = protocol['seif_type_mapping']
common = set(protocol['presidio_strict_mapping'].values())
supported = set(mapping.values())
truth = {name: {} for name in ('domain', 'entity')}
for line in (CACHE / 'predictions-first.jsonl').read_text().splitlines():
    row = json.loads(line)
    truth[row['split']][row['id']] = {tuple(s) for s in row['expected']}
versions = {v: json.loads((HERE / f'predictions-{v}.json').read_text()) for v in ('baseline', 'final')}
old_report = json.loads((PRIOR / 'report.json').read_text())
current_report = json.loads((HERE / 'report.json').read_text())
assert (HERE / 'predictions-baseline.json').read_bytes() == (PRIOR / 'predictions-final.json').read_bytes()
prior_checks = []
for split in truth:
    for section in current_report['splits'][split]['sections']:
        for profile in ('rules', 'hybrid_person_only'):
            before = current_report['splits'][split]['sections'][section]['systems']['baseline_' + profile]
            prior = old_report['splits'][split]['sections'][section]['systems']['final_' + profile]
            assert before == prior
            prior_checks.append(split + '/' + section + '/' + profile)


def characters(spans, typed=True):
    return {(kind, offset) if typed else offset for kind, start, end in spans for offset in range(start, end)}


def compare(gold, before, after):
    return {'lost_tp': len((gold & before) - after), 'gained_tp': len((gold & after) - before),
            'new_fp': len((after - before) - gold), 'removed_fp': len((before - after) - gold)}


def slice_audit(split, labels, ids, kinds, profile, name, selection):
    metric_names = ('exact_span', 'typed_character') if name.startswith('type/') else ('exact_span', 'typed_character', 'untyped_service_character')
    deltas = {key: [] for key in metric_names}
    changed_mapped = changed_service = 0
    for cid in ids:
        gold = {span for span in labels[cid] if span[0] in kinds}
        raw = [{tuple(s) for s in versions[v][split][profile][cid]} for v in ('baseline', 'final')]
        guesses = [{(mapping[kind], start, end) for kind, start, end in spans if kind in mapping and mapping[kind] in kinds}
                   for spans in raw]
        before, after = guesses
        changed_mapped += before != after
        changed_service += characters(raw[0], False) != characters(raw[1], False)
        deltas['exact_span'].append(compare(gold, before, after))
        deltas['typed_character'].append(compare(characters(gold), characters(before), characters(after)))
        # Separate privacy diagnostic: all service spans may protect a gold
        # character, even if the service entity type lacks an evaluation mapping.
        if 'untyped_service_character' in deltas:
            deltas['untyped_service_character'].append(compare(characters(gold, False), characters(raw[0], False), characters(raw[1], False)))
    aggregate = {}
    for kind, rows in deltas.items():
        aggregate[kind] = {key: sum(row[key] for row in rows) for key in ('lost_tp', 'gained_tp', 'new_fp', 'removed_fp')}
        aggregate[kind]['cases_with_lost_tp'] = sum(row['lost_tp'] > 0 for row in rows)
        aggregate[kind]['cases_with_new_fp'] = sum(row['new_fp'] > 0 for row in rows)
        aggregate[kind]['cases_with_any_regression'] = sum(row['lost_tp'] > 0 or row['new_fp'] > 0 for row in rows)
    return {'split': split, 'stratum': name, 'selection': selection, 'profile': profile, 'cases': len(ids),
            'changed_mapped_cases': changed_mapped, 'changed_service_mask_cases': None if name.startswith('type/') else changed_service, 'changes': aggregate}


slices = []
for split, labels in truth.items():
    domains = {}
    for row in pq.read_table(CACHE / f'{split}-00000-of-00001.parquet', columns=['id', 'domain']).to_pylist():
        domains.setdefault(row['domain'], []).append(row['id'])
    all_types = {span[0] for spans in labels.values() for span in spans}
    for profile in ('rules', 'hybrid_person_only'):
        for subset, kinds in (('common4', common), ('supported8', supported)):
            ids = [key for key, gold in labels.items() if all(s[0] in kinds for s in gold)]
            slices.append(slice_audit(split, labels, ids, kinds, profile, subset, 'Original whole-case subset including empty gold'))
        slices.append(slice_audit(split, labels, list(labels), all_types, profile, 'all_cases', 'Every original case and gold type; predictions use original strict mapping'))
        for domain, ids in sorted(domains.items()):
            slices.append(slice_audit(split, labels, ids, all_types, profile, 'domain/' + domain, 'Every original case in this domain, all original gold types'))
        for kind in sorted(all_types | supported):
            slices.append(slice_audit(split, labels, list(labels), {kind}, profile, 'type/' + kind, 'This gold type across all cases; predictions filtered to this type'))
report = {'schema_version': 1, 'scope': 'Aggregate paired-case audit, without corpus text output or modifications.',
          'baseline_revision': current_report['baseline_revision'], 'final_revision': current_report['final_revision'],
          'prior_4e_reproduction': {'predictions_byte_identical': True, 'primary_secondary_metrics_exact': prior_checks},
          'service_character_diagnostic': 'Unmodified actual service spans ignoring types, on whole-case and domain slices with full original gold. Not calculated per entity type: masking a different true PII type is not a service false positive. This is additional to the historical typed metric.',
          'slices': slices,
          'regression_strata': [{key: value for key, value in row.items() if key != 'selection'} for row in slices
                                if any(c['lost_tp'] or c['new_fp'] for c in row['changes'].values())]}
with (HERE / 'case-audit.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
combined = current_report | {'paired_case_audit': report}
with (HERE / 'report-with-case-audit.json').open('x') as stream:
    json.dump(combined, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
summary = {'baseline_revision': current_report['baseline_revision'], 'final_revision': current_report['final_revision'],
           'primary_secondary_metrics': json.loads((HERE / 'summary.json').read_text())['metrics'],
           'paired_case_audit_primary_secondary': [row for row in slices if row['stratum'] in ('common4', 'supported8', 'all_cases')],
           'regression_strata': report['regression_strata']}
with (HERE / 'summary-with-case-audit.json').open('x') as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
validation = json.loads((HERE / 'validation.json').read_text()) | {
    'prior_4e_reproduction': report['prior_4e_reproduction'],
    'paired_case_audit': {'slice_count': len(slices), 'artifact': 'case-audit.json', 'raw_text_output': False,
                        'source': 'Frozen expected/predicted offsets and original id/domain columns'},
    'post_measurement_targeted_diagnosis': {'prior_trial_authorized_cases': ['passport_002', 'passport_015', 'passport_048'], 'accepted_run_texts_inspected': False, 'texts_stored': False},
}
with (HERE / 'validation-with-case-audit.json').open('x') as stream:
    json.dump(validation, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
for row in slices:
    if row['stratum'] in ('common4', 'supported8', 'all_cases'):
        print(row['split'], row['stratum'], row['profile'], row['changed_mapped_cases'], row['changes'])

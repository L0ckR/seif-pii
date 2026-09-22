"""Paired per-case regression audit using cached offsets only, never example text."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path('/home/lockr/projects/seif-pii/output/external-bench')
protocol = json.loads((CACHE / 'prepared-protocol.json').read_text())
mapping = protocol['seif_type_mapping']
common = set(protocol['presidio_strict_mapping'].values())
supported = set(mapping.values())
truth = {name: {} for name in ('domain', 'entity')}
for line in (CACHE / 'predictions-first.jsonl').read_text().splitlines():
    row = json.loads(line)
    truth[row['split']][row['id']] = {tuple(s) for s in row['expected']}
versions = {v: json.loads((HERE / f'predictions-{v}.json').read_text()) for v in ('baseline', 'final')}


def typed_characters(spans):
    return {(kind, offset) for kind, start, end in spans for offset in range(start, end)}


def compare(gold, before, after):
    return {'lost_true_positives': len((gold & before) - after),
            'gained_true_positives': len((gold & after) - before),
            'new_false_positives': len((after - before) - gold),
            'removed_false_positives': len((before - after) - gold)}


report = {'scope': 'Paired per-case changes within historical whole-case subsets; offsets only, no error texts inspected.', 'slices': []}
for split, labels in truth.items():
    for subset, kinds in (('common4', common), ('supported8', supported)):
        ids = [key for key, gold in labels.items() if all(s[0] in kinds for s in gold)]
        for profile in ('rules', 'hybrid_person_only'):
            changed = []
            for key in ids:
                gold = labels[key]
                guesses = []
                for version in ('baseline', 'final'):
                    guesses.append({(mapping[kind], start, end) for kind, start, end in versions[version][split][profile][key]
                                    if kind in mapping and mapping[kind] in kinds})
                before, after = guesses
                if before != after:
                    changed.append({'case_id': key, 'exact_span': compare(gold, before, after),
                                    'typed_character': compare(typed_characters(gold), typed_characters(before), typed_characters(after))})
            regression_counts = {metric: {'cases_with_lost_tp': sum(row[metric]['lost_true_positives'] > 0 for row in changed),
                                         'cases_with_new_fp': sum(row[metric]['new_false_positives'] > 0 for row in changed),
                                         'lost_tp': sum(row[metric]['lost_true_positives'] for row in changed),
                                         'new_fp': sum(row[metric]['new_false_positives'] for row in changed)}
                                 for metric in ('exact_span', 'typed_character')}
            report['slices'].append({'split': split, 'subset': subset, 'profile': profile, 'cases': len(ids),
                                     'changed_cases': len(changed), 'regression_counts': regression_counts})
with (HERE / 'case-audit.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
for row in report['slices']:
    print(row['split'], row['subset'], row['profile'], 'changed', row['changed_cases'], row['regression_counts'])

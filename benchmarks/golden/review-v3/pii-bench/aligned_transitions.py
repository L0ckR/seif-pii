"""Whole-corpus typed and actual service privacy counts from frozen offsets."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path('/home/lockr/projects/seif-pii/output/external-bench')
protocol = json.loads((CACHE / 'prepared-protocol.json').read_text())
mapping = protocol['seif_type_mapping']
gold = {name: {} for name in ('domain', 'entity')}
for line in (CACHE / 'predictions-first.jsonl').read_text().splitlines():
    row = json.loads(line)
    gold[row['split']][row['id']] = {tuple(s) for s in row['expected']}
versions = {v: json.loads((HERE / f'predictions-{v}.json').read_text()) for v in ('baseline', 'final')}


def chars(spans, typed):
    return {(kind, i) if typed else i for kind, start, end in spans for i in range(start, end)}


def counts(truth, predicted):
    return {'tp': len(truth & predicted), 'fp': len(predicted - truth), 'fn': len(truth - predicted)}


result = {'scope': 'Every one of 1810 aligned original cases; original13 gold labels; historical strict mapping for typed metrics and ALL actual service spans for untyped privacy.', 'profiles': {}}
for profile in ('rules', 'hybrid_person_only'):
    sections = {}
    for split, rows in gold.items():
        metrics = {name: {'baseline': {'tp': 0, 'fp': 0, 'fn': 0}, 'final': {'tp': 0, 'fp': 0, 'fn': 0}}
                   for name in ('exact_span', 'typed_character', 'untyped_service_character')}
        for cid, expected in rows.items():
            for version in ('baseline', 'final'):
                raw = {tuple(s) for s in versions[version][split][profile][cid]}
                mapped = {(mapping[k], s, e) for k, s, e in raw if k in mapping}
                pairs = {'exact_span': (expected, mapped), 'typed_character': (chars(expected, True), chars(mapped, True)),
                         'untyped_service_character': (chars(expected, False), chars(raw, False))}
                for metric, (truth, predicted) in pairs.items():
                    for key, value in counts(truth, predicted).items(): metrics[metric][version][key] += value
        for m in metrics.values(): m['delta'] = {key: m['final'][key] - m['baseline'][key] for key in ('tp', 'fp', 'fn')}
        sections[split] = {'cases': len(rows), 'metrics': metrics}
    total = {}
    for metric in sections['domain']['metrics']:
        total[metric] = {version: {key: sum(section['metrics'][metric][version][key] for section in sections.values()) for key in ('tp', 'fp', 'fn')}
                         for version in ('baseline', 'final', 'delta')}
    result['profiles'][profile] = sections | {'all_aligned': {'cases': 1810, 'metrics': total}}
with (HERE / 'aligned-transitions.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')

#!/usr/bin/env python3
"""Offline code-regression replay; no network, fitting, or error-text output."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_source(source):
    return {str(p.relative_to(source)): digest(p) for directory in ('seif', 'scripts')
            for p in sorted((source / directory).glob('*.py'))}


def network_disabled(*args, **kwargs):
    raise RuntimeError('Network prohibited during frozen offline evaluation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--historical-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite prior result')
    socket.socket.connect = network_disabled
    socket.socket.connect_ex = network_disabled
    socket.create_connection = network_disabled
    sys.path.insert(0, str(args.source.resolve()))
    from scripts.evaluate_external import measure, typed_characters
    from scripts.evaluate_redmadrobot import COMMON, GOLD_MAP, SEIF_MAP, STRUCTURED, coarsen, merge_adjacent, read_rows
    from seif.detector import Span, detect, merge_ner_candidates, merge_person_candidates

    source_hashes = freeze_source(args.source)
    inputs = [args.data / filename for filename in (
        'redmadrobot-test.csv', 'redmadrobot-README.md', 'redmadrobot-protocol.json',
        'redmadrobot-predictions-first.jsonl')]
    input_hashes = {p.name: digest(p) for p in inputs}
    protocol = json.loads(inputs[2].read_text())
    historical = json.loads(args.historical_report.read_text())
    assert input_hashes['redmadrobot-test.csv'] == protocol['dataset']['file_sha256']['test.csv']
    assert input_hashes['redmadrobot-README.md'] == protocol['dataset']['file_sha256']['README.md']
    assert input_hashes['redmadrobot-predictions-first.jsonl'] == historical['raw_prediction_sha256']
    assert input_hashes['redmadrobot-protocol.json'] == historical['protocol_sha256']
    assert GOLD_MAP == protocol['mapping']['gold'] and SEIF_MAP == protocol['mapping']['seif']
    assert sorted(COMMON) == protocol['common_scope']
    rows, excluded = read_rows(args.data)
    assert excluded == protocol['dataset']['excluded_before_inference']
    assert len(rows) == protocol['dataset']['aligned_rows']
    cached = {r['id']: r for r in map(json.loads, inputs[3].read_text().splitlines())}
    assert set(cached) == {row['id'] for row in rows}
    truth, predictions = {}, {mode: {} for mode in ('rules', 'person_only', 'person_location')}
    for row in rows:
        key, text = row['id'], row['text']
        previous = cached[key]
        assert row['fine_gold'] == {tuple(item) for item in previous['fine_gold']}
        truth[key] = merge_adjacent(text, coarsen(row['fine_gold'], GOLD_MAP, keep_unknown=True))
        assert truth[key] == {tuple(item) for item in previous['merged_gold']}
        base = detect(text)
        frozen_ner = previous['original_predictions']['presidio_ru']
        assert all(0 <= start < end <= len(text) for _, start, end in frozen_ner)
        person = [Span(start, end, kind, 0.85, 'ner-person')
                  for kind, start, end in frozen_ner if kind == 'PERSON']
        person_location = [Span(start, end, kind, 0.85, 'cached-ner')
                           for kind, start, end in frozen_ner if kind in {'PERSON', 'LOCATION'}]
        outputs = {'rules': base, 'person_only': merge_person_candidates(text, base, person),
                   'person_location': merge_ner_candidates(text, base, person_location)}
        for mode, spans in outputs.items():
            mapping = {**SEIF_MAP, 'LOCATION': 'LOCATION'} if mode == 'person_location' else SEIF_MAP
            coarse = coarsen({(s.type, s.start, s.end) for s in spans}, mapping)
            predictions[mode][key] = merge_adjacent(text, coarse)

    scopes = {'common5': (COMMON, True), 'all_mapped8': (COMMON | STRUCTURED, True),
              'structured_all': (STRUCTURED, False)}
    scopes.update({kind.lower() + '_all': ({kind}, False) for kind in sorted(COMMON | STRUCTURED)})
    report_scopes = {}
    for name, (allowed, whole_cases) in scopes.items():
        ids = [key for key, gold in truth.items() if not whole_cases or all(s[0] in allowed for s in gold)]
        if name == 'common5':
            assert ids == historical['common5']['case_ids']
        gold = {key: {s for s in truth[key] if s[0] in allowed} for key in ids}
        gold_chars = typed_characters(gold)
        systems, paired_counts = {}, {}
        for mode, values in predictions.items():
            pred = {key: {s for s in values[key] if s[0] in allowed} for key in ids}
            pred_chars = typed_characters(pred)
            systems[mode] = {'typed_character': measure(gold_chars, pred_chars),
                             'merged_exact_span': measure(gold, pred)}
            for metrics in systems[mode].values():
                metrics.pop('first_five_error_offsets', None)
            paired_counts[mode] = {key: [len(gold_chars[key] & pred_chars[key]),
                                       len(pred_chars[key] - gold_chars[key]),
                                       len(gold_chars[key] - pred_chars[key])]
                                   for key in ids}
        report_scopes[name] = {'cases': len(ids), 'case_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                               'types': sorted(allowed), 'whole_cases': whole_cases, 'systems': systems,
                               'paired_character_counts': paired_counts}
    assert source_hashes == freeze_source(args.source)
    assert input_hashes == {p.name: digest(p) for p in inputs}
    result = {'revision': args.revision, 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': source_hashes, 'input_sha256': input_hashes, 'python': platform.python_version(),
              'runner_sha256': digest(Path(__file__)), 'offered_rows': protocol['dataset']['offered_rows'],
              'aligned_rows': len(rows), 'excluded_rows': excluded, 'scopes': report_scopes,
              'mapping': {'gold': GOLD_MAP, 'person_only_and_rules': SEIF_MAP,
                          'person_location': {**SEIF_MAP, 'LOCATION': 'LOCATION'}},
              'validation': {'source_and_inputs_immutable': True, 'cached_gold_matches_csv': True,
                             'frozen_protocol_exclusions_unchanged': True, 'historical_common5_ids_unchanged': True,
                             'network_disabled': True, 'live_ner_inference': False,
                             'cached_ner_score': 0.85, 'cached_ner_scores_reconstructed': True}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'revision': args.revision, 'offered_rows': result['offered_rows'], 'aligned_rows': len(rows),
                      'common5': {mode: metrics['typed_character'] for mode, metrics in report_scopes['common5']['systems'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

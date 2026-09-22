#!/usr/bin/env python3
"""Frozen Scanpatch type/offset/mask parity worker; no model inference/network."""
import argparse
import hashlib
import importlib.util
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(path):
    return {str(p.relative_to(path)): sha(p) for directory in ('seif', 'scripts')
            for p in sorted((path/directory).glob('*.py'))}


def disable_network(*args, **kwargs):
    raise RuntimeError('Network prohibited during frozen offline evaluation')


p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--revision', required=True)
p.add_argument('--support-source', type=Path, required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise RuntimeError('Refusing to overwrite existing replay')
root = Path(__file__).resolve().parent
socket.socket.connect = disable_network
socket.socket.connect_ex = disable_network
socket.create_connection = disable_network
inputs = {name: a.data/name for name in ('scanpatch-test.parquet', 'scanpatch-README.md',
          'scanpatch-protocol.json', 'scanpatch-predictions-first.jsonl')}
inputs.update({'original_evaluator.py': root/'original_evaluator.py', 'original-report.json': root/'original-report.json'})
input_hashes = {name: sha(path) for name, path in inputs.items()}
sources, support_sources = source_hashes(a.source), source_hashes(a.support_source)
sys.path.insert(0, str(a.support_source.resolve()))
spec = importlib.util.spec_from_file_location('frozen_scanpatch_support', root/'original_evaluator.py')
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)
original_report = json.loads((root/'original-report.json').read_text())
rows, cache = support.rows_and_cache(a.data, original_report)
evidence = support.score_evidence(original_report)
protocol = json.loads((a.data/'scanpatch-protocol.json').read_text())
assert protocol['mapping']['gold'] == support.GOLD_MAP
assert protocol['mapping']['seif'] == support.LOCAL_MAP
sys.path.insert(0, str(a.source.resolve()))
from seif import detector
from seif.transform import mask, restore_exact
assert Path(detector.__file__).resolve() == (a.source/'seif/detector.py').resolve()
profiles = ('rules', 'hybrid')
raw, mapped, masks, offsets = ({mode: {} for mode in profiles} for _ in range(4))
gold = {row['id']: row['merged_gold'] for row in rows}
for row in rows:
    key, text = row['id'], row['text']
    base = detector.detect(text)
    ner = [detector.Span(start, end, kind, 0.85, 'ner')
           for kind, start, end in cache[key]['original_predictions']['presidio_ru']
           if kind in {'PERSON', 'LOCATION'}]
    outputs = {'rules': base, 'hybrid': detector.merge_ner_candidates(text, base, ner)}
    for mode, spans in outputs.items():
        raw[mode][key] = [[s.type, s.start, s.end] for s in spans]
        mapped[mode][key] = support.merge_adjacent(text, support.coarsen({tuple(s) for s in raw[mode][key]}, support.LOCAL_MAP))
        masked, replacements = mask(text, spans, 'mask')
        assert len(masked) == len(text)
        assert restore_exact({'masked': masked, 'replacements': replacements}) == text
        masks[mode][key] = hashlib.sha256(masked.encode()).hexdigest()
        offsets[mode][key] = [i for i, char in enumerate(text) if char.isalnum() and masked[i] == '*']
scopes = {
    'primary_common_whole_cases': support.score_scope(gold, mapped, support.COMMON),
    'common_types_all_532_diagnostic': support.score_scope(gold, mapped, support.COMMON, whole_cases=False),
    'person_all_532_diagnostic': support.score_scope(gold, mapped, {'PERSON'}, whole_cases=False),
    'location_all_532_diagnostic': support.score_scope(gold, mapped, {'LOCATION'}, whole_cases=False),
    'document_tax_all_532_diagnostic': support.score_scope(gold, mapped, {'DOCUMENT', 'INN'}, whole_cases=False),
}
assert len(scopes['primary_common_whole_cases']['case_ids']) == 164
for scope in scopes.values():
    for scores in scope['systems'].values():
        for value in scores.values():
            value.pop('first_five_error_offsets', None)
assert sources == source_hashes(a.source)
assert support_sources == source_hashes(a.support_source)
assert input_hashes == {name: sha(path) for name, path in inputs.items()}
result = {'revision': a.revision, 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
          'cases': len(rows), 'primary_cases': 164, 'source_sha256': sources,
          'support_source_sha256': support_sources, 'input_sha256': input_hashes,
          'runner_sha256': sha(Path(__file__)), 'python': platform.python_version(),
          'score_reconstruction_evidence': evidence,
          'mapping': {'gold': support.GOLD_MAP, 'local': support.LOCAL_MAP},
          'fine_gold': {row['id']: sorted(row['fine_gold']) for row in rows},
          'mapped_gold': {key: sorted(value) for key, value in gold.items()},
          'raw_predictions': raw,
          'mapped_predictions': {mode: {key: sorted(value) for key, value in values.items()} for mode, values in mapped.items()},
          'masked_text_sha256': masks, 'actual_masked_alnum_offsets': offsets, 'scopes': scopes,
          'validation': {'network_disabled': True, 'live_model_inference': False,
                         'sources_and_inputs_immutable': True, 'all_frozen_gold_offsets_validated': True,
                         'mask_restore_all_profile_rows_passed': True}}
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'revision': a.revision, 'cases': len(rows), 'profiles': list(profiles), 'roundtrips':len(rows)*len(profiles)}))

"""Frozen ordered-output parity; never emits corpus text or invokes a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path('/home/lockr/projects/seif-pii-code-quality')
HERE = Path(__file__).resolve().parent
DATA = Path('/home/lockr/projects/seif-pii/output/external-bench')
ACCEPTED = Path('/home/lockr/projects/seif-pii-golden-improvements/output/generalized-fixes-v3/pii-bench-release')
REVISIONS = {'accepted': '12191083eea106edf85c7bbc6e60a140faf1b4c2', 'integrated': '1cde5ce3c6a70fabfe8f09b56f01de624d69633d'}


def blocked_network(*_args, **_kwargs):
    raise RuntimeError('Network is forbidden during parity replay')


socket.create_connection = blocked_network
socket.socket.connect = blocked_network
socket.socket.connect_ex = blocked_network


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def frozen_cache():
    cache = {split: {} for split in ('domain', 'entity')}
    for line in (DATA / 'predictions-first.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['id'] in cache[row['split']]:
            raise ValueError('Duplicate cache case ID')
        cache[row['split']][row['id']] = row
    return cache


def corpus():
    import pyarrow.parquet as pq
    return {split: pq.read_table(DATA / f'{split}-00000-of-00001.parquet').to_pylist()
            for split in ('domain', 'entity')}


def snapshot(version):
    root = HERE / ('source-' + version)
    revision = REVISIONS[version]
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'seif'], cwd=REPOSITORY, text=True).splitlines()
    names += ['scripts/evaluate_external.py', 'scripts/compare_presidio.py']
    manifest = {}
    for name in names:
        content = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=REPOSITORY)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        manifest[name] = sha(target)
    if version == 'accepted':
        if manifest != read(ACCEPTED / 'source-final.json')['files']:
            raise ValueError('Accepted source snapshot differs from original accepted report')
    write(HERE / ('manifest-' + version + '.json'), {'revision': revision, 'files': manifest})


def validate_sources(version):
    root = HERE / ('source-' + version)
    manifest = read(HERE / ('manifest-' + version + '.json'))
    if any(sha(root / name) != expected for name, expected in manifest['files'].items()):
        raise ValueError('Snapshot changed during parity replay')
    return manifest


def worker(version):
    validate_sources(version)
    root = HERE / ('source-' + version)
    sys.path.insert(0, str(root))
    from seif.detector import Span, detect, merge_person_candidates
    from seif.transform import mask, restore_exact
    if Path(sys.modules['seif.detector'].__file__).resolve() != (root / 'seif/detector.py').resolve():
        raise ValueError('Wrong source module imported')
    cache = frozen_cache()
    accepted = read(ACCEPTED / 'predictions-final.json')
    data = corpus()
    outputs = {}
    accepted_mismatches = []
    restoration_count = 0
    for split, rows in data.items():
        outputs[split] = {'rules': {}, 'hybrid_person_only': {}}
        for row in rows:
            text, cid = row['text'], row['id']
            base = detect(text)
            candidates = [Span(start, end, 'PERSON', 0.85, 'frozen-presidio-person')
                          for kind, start, end in cache[split][cid]['predictions']['presidio_ru'] if kind == 'PERSON']
            hybrid = merge_person_candidates(text, base, candidates)
            for profile, spans in (('rules', base), ('hybrid_person_only', hybrid)):
                masked, replacements = mask(text, spans, 'mask')
                restored = restore_exact({'masked': masked, 'replacements': replacements})
                if restored != text:
                    raise ValueError('Exact restoration failed: ' + split + '/' + cid)
                restoration_count += 1
                simple = sorted([s.type, s.start, s.end] for s in spans)
                if simple != accepted[split][profile][cid]:
                    accepted_mismatches.append({'split': split, 'profile': profile, 'id': cid})
                outputs[split][profile][cid] = {
                    'ordered_spans': [[s.start, s.end, s.type, s.confidence, s.reason] for s in spans],
                    'mask_sha256': hashlib.sha256(masked.encode()).hexdigest(),
                    'mask_characters': len(masked),
                    'replacement_record_sha256': canonical_hash(replacements),
                    'restored_sha256': hashlib.sha256(restored.encode()).hexdigest(),
                    'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                }
    validate_sources(version)
    write(HERE / ('outputs-' + version + '.json'), outputs)
    write(HERE / ('worker-' + version + '.json'), {
        'revision': REVISIONS[version], 'restoration_roundtrips': restoration_count,
        'accepted_cached_span_mismatches': accepted_mismatches,
        'output_sha256': sha(HERE / ('outputs-' + version + '.json')),
        'model_calls': 0, 'network_calls': 0, 'raw_text_outputs': False,
    })
    print(json.dumps({'worker': version, 'cases': 1810, 'profiles': 2, 'roundtrips': restoration_count,
                      'accepted_cached_span_mismatches': len(accepted_mismatches)}))


def main():
    provenance = read(ACCEPTED / 'provenance.json')
    accepted_manifest = read(ACCEPTED / 'artifact-sha256.json')
    for name in ('predictions-final.json', 'source-final.json', 'provenance.json', 'validation.json'):
        if sha(ACCEPTED / name) != accepted_manifest[name]:
            raise ValueError('Accepted artifact hash mismatch: ' + name)
    for name, expected in provenance['dataset_file_sha256'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Original dataset hash mismatch: ' + name)
    if sha(DATA / 'predictions-first.jsonl') != provenance['frozen_cache_sha256']:
        raise ValueError('Original NER cache mismatch')
    if sha(DATA / 'prepared-protocol.json') != provenance['prepared_protocol_sha256']:
        raise ValueError('Original protocol mismatch')
    if sha(ACCEPTED / 'predictions-final.json') != provenance['prediction_sha256']['final']:
        raise ValueError('Accepted prediction hash mismatch')
    data, cache = corpus(), frozen_cache()
    for split, rows in data.items():
        if len(rows) != {'domain': 900, 'entity': 910}[split] or {r['id'] for r in rows} != set(cache[split]):
            raise ValueError('Corpus/cache coverage mismatch')
        for row in rows:
            expected = sorted((e['type'], e['start'], e['end']) for e in row['entities'])
            if expected != sorted(tuple(e) for e in cache[split][row['id']]['expected']):
                raise ValueError('Original annotations changed')
            if any(row['text'][e['start']:e['end']] != e['text'] for e in row['entities']):
                raise ValueError('Original annotation substring mismatch')
    for version in REVISIONS:
        snapshot(version)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', version], cwd=REPOSITORY,
                       env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, check=True)
    before, after = [read(HERE / ('outputs-' + version + '.json')) for version in REVISIONS]
    differences = []
    summary = []
    fields = ('ordered_spans', 'mask_sha256', 'mask_characters', 'replacement_record_sha256', 'restored_sha256', 'input_sha256')
    for split in before:
        for profile in before[split]:
            counters = {field: 0 for field in fields}
            for cid, old in before[split][profile].items():
                new = after[split][profile][cid]
                changed = [field for field in fields if old[field] != new[field]]
                for field in changed:
                    counters[field] += 1
                if changed:
                    differences.append({'split': split, 'profile': profile, 'id': cid, 'changed_fields': changed,
                                        'ordered_spans_before': old['ordered_spans'], 'ordered_spans_after': new['ordered_spans']})
            summary.append({'split': split, 'profile': profile, 'cases': len(before[split][profile]), 'changed': counters})
    workers = {v: read(HERE / ('worker-' + v + '.json')) for v in REVISIONS}
    if workers['accepted']['accepted_cached_span_mismatches']:
        raise ValueError('Accepted original code failed to reproduce its accepted cached spans')
    sources = {v: validate_sources(v) for v in REVISIONS}
    if any(sha(DATA / name) != expected for name, expected in provenance['dataset_file_sha256'].items()):
        raise ValueError('Dataset changed during replay')
    passed = not differences and not workers['integrated']['accepted_cached_span_mismatches']
    report = {'schema_version': 1, 'measured_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': passed,
              'revisions': REVISIONS, 'dataset_revision': provenance['dataset_revision'], 'split_cases': {'domain': 900, 'entity': 910},
              'profiles': ['rules', 'hybrid_person_only'], 'ner_policy': {'entities': ['PERSON'], 'score': 0.85, 'source': 'same original frozen Presidio cache, no LOCATION, no model inference'},
              'ordered_span_fields': ['start', 'end', 'type', 'confidence', 'reason'], 'parity': summary,
              'changed_case_profiles': len(differences), 'differences': differences,
              'outputs_byte_identical': sha(HERE / 'outputs-accepted.json') == sha(HERE / 'outputs-integrated.json'),
              'restoration_roundtrips': sum(v['restoration_roundtrips'] for v in workers.values()),
              'accepted_original_cache_reproduced': True, 'source_snapshots_verified': True,
              'source_manifests': {v: 'manifest-' + v + '.json' for v in sources},
              'source_tree_sha256': {v: canonical_hash(s['files']) for v, s in sources.items()},
              'dataset_file_sha256': provenance['dataset_file_sha256'], 'frozen_ner_cache_sha256': provenance['frozen_cache_sha256'],
              'original_protocol_sha256': provenance['prepared_protocol_sha256'],
              'accepted_prediction_sha256': provenance['prediction_sha256']['final'],
              'worker_output_sha256': {v: w['output_sha256'] for v, w in workers.items()},
              'runtime': {'executable': sys.executable, 'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in ('pyarrow', 'numpy')}},
              'runner_sha256': sha(Path(__file__)), 'model_calls': 0, 'network_calls': 0, 'raw_corpus_texts_written': False,
              'limitations': ['Parity on this frozen corpus/profile, not a proof for every possible input.',
                              'PERSON-only historical NER cache; not deployed PERSON+LOCATION or HTTP/RPS.',
                              'Actual mask and replacement-record equality are verified with SHA256; original texts and replacement originals remain in memory only.']}
    write(HERE / 'report.json', report)
    write(HERE / 'summary.json', {k: report[k] for k in ('passed', 'revisions', 'dataset_revision', 'split_cases', 'profiles', 'parity', 'changed_case_profiles', 'outputs_byte_identical', 'restoration_roundtrips', 'accepted_original_cache_reproduced', 'source_tree_sha256', 'limitations')})
    write(HERE / 'artifact-sha256.json', {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'artifact-sha256.json'})
    print(json.dumps({'passed': passed, 'changed_case_profiles': len(differences), 'outputs_byte_identical': report['outputs_byte_identical'], 'roundtrips': report['restoration_roundtrips']}))
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=REVISIONS)
    options = parser.parse_args()
    if options.worker:
        worker(options.worker)
    else:
        raise SystemExit(main())

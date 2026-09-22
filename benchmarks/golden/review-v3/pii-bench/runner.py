#!/usr/bin/env python3
"""Offline, frozen PII-Bench regression. No dataset examples or error texts emitted."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TASK_ROOT = Path('/home/lockr/projects/seif-pii-golden-improvements')
BASELINE_ROOT = TASK_ROOT / 'output/generalized-fixes-v3/baseline-source'
CACHE_ROOT = Path('/home/lockr/projects/seif-pii/output/external-bench')
OUT = TASK_ROOT / 'output/generalized-fixes-v3/pii-bench-release'
REVISIONS = {'baseline': '4e46c86a6f6c7bf91e9525968981ae47d7984aba', 'final': '12191083eea106edf85c7bbc6e60a140faf1b4c2'}
ROOTS = {'baseline': BASELINE_ROOT, 'final': TASK_ROOT}
SCORE = 0.85


def deny_network(*_args, **_kwargs):
    raise RuntimeError('Network calls forbidden during frozen evaluation')


socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def evaluate_module(root):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('frozen_external_evaluator', root / 'scripts/evaluate_external.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows():
    import pyarrow.parquet as pq
    return {split: pq.read_table(CACHE_ROOT / f'{split}-00000-of-00001.parquet').to_pylist()
            for split in ('domain', 'entity')}


def load_cache():
    result = {split: {} for split in ('domain', 'entity')}
    for line in (CACHE_ROOT / 'predictions-first.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['id'] in result[row['split']]:
            raise ValueError('Duplicate frozen prediction ID')
        result[row['split']][row['id']] = row
    return result


def source_hashes(root, revision):
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'seif'], cwd=TASK_ROOT, text=True).splitlines()
    names += ['scripts/evaluate_external.py', 'scripts/compare_presidio.py']
    hashes = {}
    for name in names:
        expected = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=TASK_ROOT)
        path = root / name
        if path.read_bytes() != expected:
            raise ValueError('Source differs from pinned Git revision: ' + name)
        hashes[name] = digest(path)
    actual_python = {str(p.relative_to(root)) for p in (root / 'seif').rglob('*.py')}
    if actual_python != {n for n in names if n.startswith('seif/') and n.endswith('.py')}:
        raise ValueError('Unexpected source modules in pinned snapshot')
    return hashes


def worker(version):
    root = ROOTS[version]
    before = source_hashes(root, REVISIONS[version])
    sys.path.insert(0, str(root))
    from seif.detector import Span, detect, merge_person_candidates
    from seif.transform import mask, restore_exact
    if Path(sys.modules['seif.detector'].__file__).resolve() != (root / 'seif/detector.py').resolve():
        raise ValueError('Imported wrong source tree')
    data, cache = load_rows(), load_cache()
    predictions = {}
    for split, rows in data.items():
        predictions[split] = {'rules': {}, 'hybrid_person_only': {}}
        for row in rows:
            text, case_id = row['text'], row['id']
            base = detect(text)
            person = [Span(start, end, 'PERSON', SCORE, 'frozen-presidio-person')
                      for kind, start, end in cache[split][case_id]['predictions']['presidio_ru'] if kind == 'PERSON']
            hybrid = merge_person_candidates(text, base, person)
            for name, spans in (('rules', base), ('hybrid_person_only', hybrid)):
                masked, replacements = mask(text, spans, 'mask')
                if restore_exact({'masked': masked, 'replacements': replacements}) != text:
                    raise ValueError('Exact restoration mismatch')
                predictions[split][name][case_id] = sorted((s.type, s.start, s.end) for s in spans)
    if source_hashes(root, REVISIONS[version]) != before:
        raise ValueError('Source changed during inference')
    write_json(OUT / f'predictions-{version}.json', predictions)
    write_json(OUT / f'source-{version}.json', {'revision': REVISIONS[version], 'files': before,
                                               'restoration_roundtrips': sum(len(rows) for rows in data.values()) * 2})
    print(json.dumps({'worker': version, 'cases': 1810, 'profiles': 2, 'model_calls': 0, 'network_calls': 0}))


def safe_metrics(value):
    if isinstance(value, dict):
        return {k: safe_metrics(v) for k, v in value.items() if k != 'first_five_error_offsets'}
    if isinstance(value, list):
        return [safe_metrics(v) for v in value]
    return value


def selected_report(evaluator, truth, mapped, kinds):
    return safe_metrics(evaluator.subset_metric(truth, mapped, kinds))


def validate_inputs(evaluator, data, cache, historical):
    checked = {}
    for filename, expected in evaluator.FILES.items():
        actual = digest(CACHE_ROOT / filename)
        if actual != expected or historical['dataset']['file_sha256'][filename] != expected:
            raise ValueError('Frozen dataset checksum mismatch')
        checked[filename] = actual
    marker = read_json(CACHE_ROOT / 'first-inference.json')
    protocol = read_json(CACHE_ROOT / 'prepared-protocol.json')
    protocol_sha = digest(CACHE_ROOT / 'prepared-protocol.json')
    if protocol_sha != marker['protocol_sha256'] or protocol_sha != historical['protocol_sha256']:
        raise ValueError('Prepared protocol mismatch')
    if digest(CACHE_ROOT / 'predictions-first.jsonl') != historical['raw_prediction_sha256']:
        raise ValueError('Frozen prediction checksum mismatch')
    if marker['detector_sha256'] != historical['source_hashes']['detector']:
        raise ValueError('Historical implementation metadata mismatch')
    for key, expected in (('seif_type_mapping', evaluator.SEIF_MAP), ('presidio_strict_mapping', evaluator.PRESIDIO_MAP)):
        if protocol[key] != expected:
            raise ValueError('Prepared type mapping mismatch')
    if evaluator.REVISION != historical['dataset']['revision'] or evaluator.DATASET != historical['dataset']['repository']:
        raise ValueError('Dataset revision mismatch')
    validation = {}
    for split, rows in data.items():
        if len(rows) != {'domain': 900, 'entity': 910}[split] or {r['id'] for r in rows} != set(cache[split]):
            raise ValueError('Case coverage mismatch')
        entities = 0
        for row in rows:
            truth = []
            for entity in row['entities']:
                start, end = entity['start'], entity['end']
                if entity['type'] not in evaluator.ALL_TYPES or not 0 <= start < end <= len(row['text']):
                    raise ValueError('Invalid frozen annotation')
                if row['text'][start:end] != entity['text']:
                    raise ValueError('Original annotation substring mismatch')
                truth.append((entity['type'], start, end))
                entities += 1
            if sorted(truth) != sorted(tuple(v) for v in cache[split][row['id']]['expected']):
                raise ValueError('Frozen predictions do not match original gold labels')
            for predictions in cache[split][row['id']]['predictions'].values():
                if any(not 0 <= start < end <= len(row['text']) for _, start, end in predictions):
                    raise ValueError('Cached prediction out of bounds')
        validation[split] = {'rows': len(rows), 'negative_rows': sum(not row['entities'] for row in rows), 'entities': entities,
                             'unaltered_labels': True, 'all_substrings_validated': True}
    # Cached spans omit confidence. Confirm the installed pinned spaCy adapter's
    # constant PERSON score and no PERSON context words; never instantiate a model.
    packages = Path(sys.prefix) / 'lib/python3.13/site-packages/presidio_analyzer'
    config_path = packages / 'nlp_engine/ner_model_configuration.py'
    spacy_path = packages / 'nlp_engine/spacy_nlp_engine.py'
    config = config_path.read_text()
    if ('default=0.85' not in config or 'LOW_SCORE_ENTITY_NAMES = set()' not in config
            or 'scores = [self.ner_model_configuration.default_score] * len(entities)' not in spacy_path.read_text()):
        raise ValueError('Cannot validate cached PERSON confidence reconstruction')
    spacy_recognizer = next(r for r in historical['presidio_configuration']['recognizers'] if r['name'] == 'SpacyRecognizer')
    if spacy_recognizer['context'] or historical['presidio_configuration']['nlp_engine']['ner_model_configuration'].get('default_score', SCORE) != SCORE:
        raise ValueError('Unexpected historic NER score configuration')
    return {'dataset': validation, 'file_sha256': checked, 'protocol_sha256': protocol_sha,
            'cache_sha256': historical['raw_prediction_sha256'], 'historical_detector_sha256': historical['source_hashes']['detector'],
            'current_evaluator_sha256': digest(TASK_ROOT / 'scripts/evaluate_external.py'),
            'historical_evaluator_sha256': historical['source_hashes']['evaluator'],
            'score_evidence': {'default': SCORE, 'score_missing_from_cache': True, 'person_context': [],
                               'configuration_sha256': digest(config_path), 'spacy_engine_sha256': digest(spacy_path)},
            'network_blocked': True, 'model_calls': 0, 'raw_error_texts_emitted': False}


def bootstrap_delta(evaluator, truth, mapped, domains, kinds):
    import numpy as np
    rng = np.random.default_rng(20260922)
    sums = {key: np.zeros((2000, 3), dtype=np.int64) for key in mapped if key != 'presidio_ru'}
    for domain_ids in domains.values():
        ids = [key for key in domain_ids if all(span[0] in kinds for span in truth[key])]
        if not ids:
            continue
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=2000)
        for system in sums:
            counts = []
            for key in ids:
                actual = {span for span in mapped[system][key] if span[0] in kinds}
                gold = truth[key]
                counts.append((len(gold & actual), len(actual - gold), len(gold - actual)))
            sums[system] += weights @ np.array(counts, dtype=np.int64)
    f1 = {key: np.divide(2 * c[:, 0], 2 * c[:, 0] + c[:, 1] + c[:, 2],
                        out=np.zeros(2000), where=(2 * c[:, 0] + c[:, 1] + c[:, 2]) != 0) for key, c in sums.items()}
    return {profile: {'f1_delta_interval95': [round(float(x), 6) for x in np.percentile(
                        f1['final_' + profile] - f1['baseline_' + profile], [2.5, 97.5])],
                      'bootstrap_fraction_delta_below_zero': round(float(np.mean(f1['final_' + profile] < f1['baseline_' + profile])), 6)}
            for profile in ('rules', 'hybrid_person_only')}


def delta_rows(section, path):
    result = []
    for profile in ('rules', 'hybrid_person_only'):
        for metric_kind in ('exact_span', 'typed_character'):
            before = section['systems']['baseline_' + profile][metric_kind]
            after = section['systems']['final_' + profile][metric_kind]
            if not before['cases']:
                continue
            for name in ('precision', 'recall', 'f1', 'f2', 'exact_cases', 'negative_cases_with_fp'):
                delta = round(after[name] - before[name], 6)
                regression = delta > 0 if name == 'negative_cases_with_fp' else delta < 0
                if regression:
                    result.append({'slice': path, 'profile': profile, 'metric_kind': metric_kind,
                                   'metric': name, 'before': before[name], 'after': after[name], 'delta': delta,
                                   'cases': before['cases']})
    return result


def main():
    evaluator = evaluate_module(TASK_ROOT)
    data, cache = load_rows(), load_cache()
    historical = read_json(TASK_ROOT / 'docs/pii-bench-comparison.json')
    validation = validate_inputs(evaluator, data, cache, historical)
    for version in ROOTS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', version, '--output', str(OUT)], check=True,
                       cwd=ROOTS[version], env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    versions = {version: read_json(OUT / f'predictions-{version}.json') for version in ROOTS}
    reports, regressions, historical_checks = {}, [], {}
    for split, rows in data.items():
        truth = {row['id']: {(item['type'], item['start'], item['end']) for item in row['entities']} for row in rows}
        domains = {}
        for row in rows:
            domains.setdefault(row['domain'], []).append(row['id'])
        cached = {system: {key: {tuple(s) for s in entry['predictions'][system]} for key, entry in cache[split].items()}
                  for system in ('seif_fast', 'seif_hybrid', 'presidio_ru')}
        old_mapped = {system: evaluator.map_predictions(predictions, evaluator.PRESIDIO_MAP if system == 'presidio_ru' else evaluator.SEIF_MAP)
                      for system, predictions in cached.items()}
        for section, kinds in (('common4_primary', evaluator.COMMON), ('supported8_requirement_coverage', evaluator.SUPPORTED)):
            old = selected_report(evaluator, truth, old_mapped, kinds)
            if old != safe_metrics(historical['splits'][split][section]):
                raise ValueError('Historical cached report failed exact reproduction: ' + split + '/' + section)
            historical_checks[split + '/' + section] = 'all systems, exact-span and typed-character results reproduced exactly'
        mapped = {'presidio_ru': old_mapped['presidio_ru']}
        for version, predictions in versions.items():
            for profile, values in predictions[split].items():
                mapped[version + '_' + profile] = evaluator.map_predictions(
                    {key: {tuple(span) for span in spans} for key, spans in values.items()}, evaluator.SEIF_MAP)
        reports[split] = {'total_cases': len(rows), 'gold_counts': dict(Counter(e['type'] for r in rows for e in r['entities'])),
                          'sections': {}, 'by_domain_common4': {}, 'by_type_all_cases': {}}
        for section, kinds in (('common4_primary', evaluator.COMMON), ('supported8_requirement_coverage', evaluator.SUPPORTED)):
            report = selected_report(evaluator, truth, mapped, kinds)
            report['paired_baseline_final_bootstrap'] = bootstrap_delta(evaluator, truth, mapped, domains, kinds)
            reports[split]['sections'][section] = report
            regressions.extend(delta_rows(report, split + '/' + section))
        for domain, ids in sorted(domains.items()):
            report = selected_report(evaluator, {key: truth[key] for key in ids},
                                     {system: {key: values[key] for key in ids} for system, values in mapped.items()}, evaluator.COMMON)
            reports[split]['by_domain_common4'][domain] = report
            regressions.extend(delta_rows(report, split + '/domain/' + domain + '/common4'))
        for kind in sorted(evaluator.SUPPORTED):
            gold = {key: {span for span in spans if span[0] == kind} for key, spans in truth.items()}
            systems = {}
            for system, values in mapped.items():
                predicted = {key: {span for span in spans if span[0] == kind} for key, spans in values.items()}
                systems[system] = {'exact_span': safe_metrics(evaluator.measure(gold, predicted)),
                                   'typed_character': safe_metrics(evaluator.measure(evaluator.typed_characters(gold), evaluator.typed_characters(predicted), False))}
            report = {'systems': systems}
            reports[split]['by_type_all_cases'][kind] = report
            regressions.extend(delta_rows(report, split + '/all_cases/type/' + kind))
    validation['historical_cache_reproduction'] = historical_checks
    validation['sources'] = {version: read_json(OUT / f'source-{version}.json') for version in ROOTS}
    validation['final_source_unchanged'] = source_hashes(TASK_ROOT, REVISIONS['final']) == validation['sources']['final']['files']
    final_report = {'schema_version': 1, 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
                    'baseline_revision': REVISIONS['baseline'], 'final_revision': REVISIONS['final'],
                    'dataset_revision': evaluator.REVISION,
                    'ner_profile': {'entities': ['PERSON'], 'score': SCORE, 'mode': 'same frozen Presidio outputs for both revisions; no NER rerun',
                                    'location': 'excluded to preserve historical protocol; this is not a PERSON+LOCATION deployment evaluation'},
                    'evaluation_status': 'offline frozen regression on previously evaluated external synthetic dataset; no tuning',
                    'splits': reports, 'regressions': regressions,
                    'limitations': ['Synthetic external dataset; this is not real organizer traffic or an unseen blind benchmark.',
                                    'Typed-character metric counts every character inside labeled spans, as in the historical protocol; organizer golden uses a different mask-character metric.',
                                    'Whole-case common4 domain subset is primary; supported8 and entity split are secondary; originals are unmodified.',
                                    'Bootstrap treats cases as independent within domain; synthetic template dependence is not modeled.',
                                    'No model, network, threshold, detector, or annotation changes were made for this run.',
                                    'PERSON confidence is reconstructed from pinned Presidio constant 0.85; the old cache did not store confidence.',
                                    'This report does not measure HTTP latency, throughput, capture overhead or the deployed PERSON+LOCATION NER profile.']}
    write_json(OUT / 'report.json', final_report)
    write_json(OUT / 'validation.json', validation)
    summary = {'revisions': REVISIONS, 'ner_profile': final_report['ner_profile'], 'metrics': [], 'regressions': regressions}
    for split, value in reports.items():
        for section, metrics in value['sections'].items():
            for profile in ('rules', 'hybrid_person_only'):
                summary['metrics'].append({'split': split, 'section': section, 'profile': profile,
                                           'before': metrics['systems']['baseline_' + profile],
                                           'after': metrics['systems']['final_' + profile],
                                           'bootstrap': metrics['paired_baseline_final_bootstrap'][profile]})
    write_json(OUT / 'summary.json', summary)
    print(json.dumps({'completed': True, 'regression_items': len(regressions), 'output': str(OUT)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=ROOTS)
    parser.add_argument('--output', type=Path, default=OUT, help='New immutable output directory for reproduction')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    worker(args.worker) if args.worker else main()

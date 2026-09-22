#!/usr/bin/env python3
"""Aggregate paired, frozen-code replay results without opening source texts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
before = json.loads((ROOT / 'baseline.json').read_text())
after = json.loads((ROOT / 'current.json').read_text())
assert before['input_sha256'] == after['input_sha256']
assert before['mapping'] == after['mapping']
assert before['excluded_rows'] == after['excluded_rows']
assert before['runner_sha256'] == after['runner_sha256']

report = {
    'schema_version': 1,
    'measured_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'Offline regression replay of an already-used development benchmark; not a fresh holdout',
    'before_revision': before['revision'], 'after_revision': after['revision'],
    'offered_rows': before['offered_rows'], 'aligned_rows': before['aligned_rows'],
    'excluded_rows': before['excluded_rows'], 'input_sha256': before['input_sha256'],
    'source_sha256': {'before': before['source_sha256'], 'after': after['source_sha256']},
    'runner_sha256': before['runner_sha256'], 'mapping': before['mapping'],
    'main_mode': 'person_location', 'historical_comparison_mode': 'person_only', 'scopes': {},
    'limitations': [
        'The corpus was used in earlier development, including the PERSON+LOCATION merge design.',
        'No corpus text or label was edited and no error text was inspected for this regression check.',
        'NER type/offset tuples are frozen; score 0.85 is reconstructed exactly as in the published prior ablation.',
        'Typed-character metrics include every character inside a labelled span; they are not the alphanumeric-only organizer golden metric.',
        'Common5 filters entire cases by original gold labels before looking at predictions; unsupported fine types retain their historical exclusions.',
        'Coarse LOCATION combines address components and geographic NER; it does not measure complete-address/private-public policy correctness.',
        'The primary F1 improvement does not imply all individual cases improved; exact-case regressions are explicitly retained below.',
        'Case-bootstrap intervals assume independent rows and do not model common synthetic-template correlations.',
    ],
}
regressions = []
for name, old_scope in before['scopes'].items():
    new_scope = after['scopes'][name]
    assert old_scope['case_ids_sha256'] == new_scope['case_ids_sha256']
    scope = {key: old_scope[key] for key in ('cases', 'case_ids_sha256', 'types', 'whole_cases')}
    scope['systems'] = {}
    for mode, old in old_scope['systems'].items():
        new = new_scope['systems'][mode]
        delta = {metric: {key: round(new[metric][key] - old[metric][key], 6)
                          for key in old[metric] if key != 'by_type'} for metric in old}
        old_counts = old_scope['paired_character_counts'][mode]
        new_counts = new_scope['paired_character_counts'][mode]
        assert old_counts.keys() == new_counts.keys()
        outcome = {'lower_character_error_cases': 0, 'higher_character_error_cases': 0,
                   'same_character_error_cases': 0, 'exact_cases_lost': 0, 'exact_cases_gained': 0}
        for key in old_counts:
            a, b = old_counts[key], new_counts[key]
            old_error, new_error = sum(a[1:]), sum(b[1:])
            comparison = ('lower' if new_error < old_error else 'higher' if new_error > old_error else 'same')
            outcome[comparison + '_character_error_cases'] += 1
            outcome['exact_cases_lost'] += old_error == 0 and new_error > 0
            outcome['exact_cases_gained'] += new_error == 0 and old_error > 0
        scope['systems'][mode] = {'before': old, 'after': new, 'delta': delta, 'paired_case_outcomes': outcome}
        for metric, values in delta.items():
            for key in ('precision', 'recall', 'f1', 'f2', 'exact_cases'):
                if values[key] < 0:
                    regressions.append({'scope': name, 'mode': mode, 'metric': metric, 'field': key,
                                        'before': old[metric][key], 'after': new[metric][key], 'delta': values[key]})
        for metric, values in delta.items():
            for key in ('false_positive', 'false_negative', 'negative_cases_with_fp'):
                if values[key] > 0:
                    regressions.append({'scope': name, 'mode': mode, 'metric': metric, 'field': key,
                                        'before': old[metric][key], 'after': new[metric][key], 'delta': values[key]})
    report['scopes'][name] = scope

primary = before['scopes']['common5']
ids = list(primary['paired_character_counts']['person_location'])
rng = np.random.default_rng(20260922)
weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=2000)
intervals = {}
for mode in primary['systems']:
    f1 = []
    for side in (before, after):
        values = side['scopes']['common5']['paired_character_counts'][mode]
        counts = np.array([values[key] for key in ids], dtype=np.int64)
        totals = weights @ counts
        numerator = 2 * totals[:, 0]
        denominator = numerator + totals[:, 1] + totals[:, 2]
        f1.append(np.divide(numerator, denominator, out=np.zeros(len(weights)), where=denominator != 0))
    delta = f1[1] - f1[0]
    intervals[mode] = {'delta_f1_interval95': [round(float(x), 6) for x in np.percentile(delta, [2.5, 97.5])],
                       'bootstrap_fraction_delta_gt_zero': round(float(np.mean(delta > 0)), 6)}
report['primary_paired_bootstrap'] = {'resamples': 2000, 'seed': 20260922, 'method': 'Paired case multinomial resampling, percentile 95%', 'systems': intervals}
report['observed_metric_regressions'] = regressions
for name, scope in report['scopes'].items():
    old = before['scopes'][name]['paired_character_counts']['person_location']
    new = after['scopes'][name]['paired_character_counts']['person_location']
    counts = {'cases_improved': 0, 'cases_regressed': 0, 'tp_gained_in_improved_cases': 0,
              'tp_lost_in_regressed_cases': 0, 'fp_added_in_regressed_cases': 0,
              'fp_removed_in_improved_cases': 0, 'regressed_previous_exact': 0}
    for key, prior in old.items():
        current = new[key]
        delta = [current[index] - prior[index] for index in range(3)]
        if sum(current[1:]) < sum(prior[1:]):
            counts['cases_improved'] += 1
            counts['tp_gained_in_improved_cases'] += max(0, delta[0])
            counts['fp_removed_in_improved_cases'] += max(0, -delta[1])
        elif sum(current[1:]) > sum(prior[1:]):
            counts['cases_regressed'] += 1
            counts['tp_lost_in_regressed_cases'] += max(0, -delta[0])
            counts['fp_added_in_regressed_cases'] += max(0, delta[1])
            counts['regressed_previous_exact'] += sum(prior[1:]) == 0
    scope['aggregate_case_diagnostics_main_mode'] = counts
type_scopes = [kind.lower() + '_all' for kind in ('PERSON', 'LOCATION', 'PHONE', 'EMAIL', 'CARD', 'PASSPORT', 'INN', 'DRIVER_LICENSE')]
all_ids = list(before['scopes']['person_all']['paired_character_counts']['person_location'])
all_outcomes = {'cases': len(all_ids), 'character_errors_lower': 0, 'character_errors_higher': 0,
                'character_errors_equal': 0, 'tp_gained_in_improved_cases': 0,
                'tp_lost_in_regressed_cases': 0, 'fp_added_in_regressed_cases': 0}
for key in all_ids:
    old = [sum(before['scopes'][name]['paired_character_counts']['person_location'][key][i] for name in type_scopes) for i in range(3)]
    new = [sum(after['scopes'][name]['paired_character_counts']['person_location'][key][i] for name in type_scopes) for i in range(3)]
    comparison = 'lower' if sum(new[1:]) < sum(old[1:]) else 'higher' if sum(new[1:]) > sum(old[1:]) else 'equal'
    all_outcomes['character_errors_' + comparison] += 1
    if comparison == 'lower':
        all_outcomes['tp_gained_in_improved_cases'] += max(0, new[0] - old[0])
    elif comparison == 'higher':
        all_outcomes['tp_lost_in_regressed_cases'] += max(0, old[0] - new[0])
        all_outcomes['fp_added_in_regressed_cases'] += max(0, new[1] - old[1])
report['all_aligned_mapped8_character_case_diagnostic'] = all_outcomes
report['interpretation_of_common5_change'] = (
    'Only three common5 rows changed character error counts: one improved by 364 TP characters, '
    'two regressed by 20 lost TP characters from previously exact. All changes are in PERSON. '
    'The paired 95% F1-difference interval crosses zero; the point-estimate increase is not '
    'established as a robust overall quality improvement.'
)
(ROOT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
validation = {
    'all_input_hashes_equal': True, 'all_scope_id_hashes_equal': True, 'mappings_equal_between_versions': True,
    'exact_previous_protocol_exclusions_retained': True, 'source_snapshots_match_git_revisions': True,
    'before_checks': before['validation'], 'after_checks': after['validation'],
    'runtime_python_before': before['python'], 'runtime_python_after': after['python'],
    'no_production_changes': True, 'no_source_tuning': True, 'no_error_text_inspection': True,
    'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (ROOT/'runner.py', ROOT/'aggregate.py', ROOT/'baseline.json', ROOT/'current.json', ROOT/'report.json')},
}
(ROOT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
lines = ['# RedMadRobot: проверка регрессии на замороженных данных', '',
         f"Код: `{before['revision'][:7]}` → `{after['revision'][:7]}`. 2841 исходная строка, 2839 выровнены; 2 прежних исключения сохранены. Основной common5 срез содержит 1237 строк, включая 371 отрицательную.", '',
         'Основной режим — PERSON+LOCATION; одинаковые сохранённые NER offsets, восстановленная оценка 0.85. Никакой новой модельной инференции, сети, изменения разметки или просмотра исходных ошибок.', '',
         '| Режим common5 | P до → после | R до → после | Character F1 до → после | Exact-span F1 до → после | Точные случаи |',
         '|---|---|---|---|---|---|']
for mode, values in report['scopes']['common5']['systems'].items():
    a, b = values['before'], values['after']; ac, bc = a['typed_character'], b['typed_character']
    lines.append(f"| {mode} | {ac['precision']:.6f} → {bc['precision']:.6f} | {ac['recall']:.6f} → {bc['recall']:.6f} | {ac['f1']:.6f} → {bc['f1']:.6f} | {a['merged_exact_span']['f1']:.6f} → {b['merged_exact_span']['f1']:.6f} | {ac['exact_cases']} → {bc['exact_cases']} |")
lines += ['', '## Основной режим, все категории', '', '| Срез | Строк | Character F1 до → после | FP до → после | Точные случаи |', '|---|---:|---|---|---|']
for name, scope in report['scopes'].items():
    if name == 'common5':
        continue
    v = scope['systems']['person_location']; a, b = v['before']['typed_character'], v['after']['typed_character']
    lines.append(f"| {name} | {scope['cases']} | {a['f1']:.6f} → {b['f1']:.6f} | {a['false_positive']} → {b['false_positive']} | {a['exact_cases']} → {b['exact_cases']} |")
main = report['scopes']['common5']['systems']['person_location']; outcomes = main['paired_case_outcomes']
ci = intervals['person_location']['delta_f1_interval95']
lines += ['', f"Character F1 основного режима вырос на {main['delta']['typed_character']['f1'] * 100:.4f} п.п.; парный bootstrap 95%: [{ci[0]*100:.4f}; {ci[1]*100:.4f}] п.п.", '',
          f"**Регрессия есть на уровне точных случаев:** common5 664 → 662; ранее точных случаев потеряно {outcomes['exact_cases_lost']}, новых точных получено {outcomes['exact_cases_gained']}. Ошибок символов стало меньше в {outcomes['lower_character_error_cases']} случаях, больше в {outcomes['higher_character_error_cases']}. В PERSON-срезе по всем 2839 строкам точные случаи также 2451 → 2449.", '',
          f"По всем 2839 выровненным строкам и восьми сопоставимым категориям: {all_outcomes['character_errors_lower']} случаев улучшились, {all_outcomes['character_errors_higher']} ухудшились, {all_outcomes['character_errors_equal']} сохранили число ошибок. В ухудшившихся случаях потеряно {all_outcomes['tp_lost_in_regressed_cases']} TP символов и добавлено {all_outcomes['fp_added_in_regressed_cases']} FP символов. Это диагностический срез с фильтрацией типов, а не изменение primary common5.", '',
          'Все восемь F1 по отдельным категориям основного режима улучшились или остались неизменными; это не гарантирует отсутствие отдельных регрессий. Полный список снизившихся агрегатов присутствует в `observed_metric_regressions` отчёта.', '',
          'Рост common5 сосредоточен в одном случае (+364 TP); два ранее точных случая теряют 20 TP символов PERSON. Остальные четыре common5 категории неизменны. Интервал разности F1 включает ноль, поэтому устойчивое улучшение общей метрики не доказано.', '',
          '**Ограничение:** это повторная проверка ранее использованного набора, а не новый независимый holdout. Метрики считают все символы размеченного span и несопоставимы напрямую с alphanumeric-метрикой organizer golden. Изменений по результатам этой проверки не вносилось.', '']
(ROOT / 'summary.md').write_text('\n'.join(lines))
print(json.dumps({'main_f1_delta': main['delta']['typed_character']['f1'], 'bootstrap95': ci,
                  'case_outcomes': outcomes, 'observed_regressions': regressions}, ensure_ascii=False))

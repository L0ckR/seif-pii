#!/usr/bin/env python3
"""Aggregate frozen numeric reports without corpus/model access (stdlib only)."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    path = ROOT / name
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def characters(spans):
    return {(kind, offset) for kind, start, end in spans for offset in range(start, end)}


def triple(gold, pred):
    return [len(gold & pred), len(pred - gold), len(gold - pred)]


before, after, prior = read('baseline.json'), read('current.json'), read('prior-reference.json')
for field in ('input_sha256', 'mapping', 'excluded_rows', 'runner_sha256'):
    assert before[field] == after[field], field
assert before['revision'] == prior['previous_after_revision']
assert digest(before['scopes']) == prior['previous_after_scopes_sha256']
assert before['paired_mapped_spans']['gold'] == after['paired_mapped_spans']['gold']
report = {
    'schema_version': 2,
    'status': 'Frozen offline regression trial; development corpus, not fresh holdout',
    'before_revision': before['revision'], 'after_revision': after['revision'],
    'measured_at_utc': {'before': before['measured_at_utc'], 'after': after['measured_at_utc']},
    'offered_rows': before['offered_rows'], 'aligned_rows': before['aligned_rows'],
    'excluded_rows': before['excluded_rows'], 'input_sha256': before['input_sha256'],
    'source_sha256': {'before': before['source_sha256'], 'after': after['source_sha256']},
    'mapping': before['mapping'], 'runner_sha256': before['runner_sha256'],
    'main_mode': 'person_location', 'historical_comparison_mode': 'person_only',
    'baseline_exactly_reproduces_previous_after_scopes': True,
    'scopes': {}, 'observed_metric_regressions': [],
    'limitations': [
        'Already-used development benchmark, including diagnostic review; not an independent holdout.',
        'Frozen NER type/offset tuples replayed at reconstructed score 0.85; no inference, network, or label changes.',
        'Typed characters include spaces/punctuation within spans; not organizer alphanumeric-only metrics.',
        'Common5 and all_mapped8 exclude whole cases with unsupported gold types before looking at predictions.',
        'All-aligned per-type/position diagnostics filter types, not cases; distinct from primary common5 estimand.',
        'No new confidence interval or statistical-significance claim is calculated.',
        'Per-position gates retain lost TP and newly added FP even when aggregate or row total errors improve.',
    ],
}
for name, scope in before['scopes'].items():
    current = after['scopes'][name]
    assert scope['case_ids_sha256'] == current['case_ids_sha256']
    result = {k: scope[k] for k in ('cases', 'case_ids_sha256', 'types', 'whole_cases')}
    result['systems'] = {}
    for mode, old in scope['systems'].items():
        new = current['systems'][mode]
        delta = {metric: {k: round(new[metric][k] - value, 6) for k, value in values.items() if k != 'by_type'}
                 for metric, values in old.items()}
        outcomes = {'lower_character_error_cases': 0, 'higher_character_error_cases': 0,
                    'same_character_error_cases': 0, 'exact_cases_lost': 0, 'exact_cases_gained': 0}
        for key, a in scope['paired_character_counts'][mode].items():
            b = current['paired_character_counts'][mode][key]
            x, y = sum(a[1:]), sum(b[1:])
            comparison = 'lower' if y < x else 'higher' if y > x else 'same'
            outcomes[comparison + '_character_error_cases'] += 1
            outcomes['exact_cases_lost'] += x == 0 and y > 0
            outcomes['exact_cases_gained'] += x > 0 and y == 0
        result['systems'][mode] = {'before': old, 'after': new, 'delta': delta, 'paired_case_outcomes': outcomes}
        for metric, changes in delta.items():
            for field, change in changes.items():
                worse = change < 0 if field in {'precision', 'recall', 'f1', 'f2', 'exact_cases'} else change > 0 if field in {'false_positive', 'false_negative', 'negative_cases_with_fp'} else False
                if worse:
                    report['observed_metric_regressions'].append({'scope': name, 'mode': mode, 'metric': metric,
                        'field': field, 'before': old[metric][field], 'after': new[metric][field], 'delta': change})
    report['scopes'][name] = result

modes = list(before['scopes']['common5']['systems'])
all_positions, closure, untyped_positions, paired_counts = {}, {}, {}, {}
for mode in modes:
    aggregate = {'cases': before['aligned_rows'], 'changed_cases': 0, 'character_errors_lower': 0,
        'character_errors_higher': 0, 'character_errors_equal': 0, 'gained_tp': 0, 'lost_tp': 0,
        'new_fp': 0, 'removed_fp': 0, 'cases_with_lost_tp_or_new_fp': 0, 'flagged_cases': [], 'by_type': {}}
    cases = {}
    paired_counts[mode] = {}
    untyped = {'cases': before['aligned_rows'], 'gained_tp': 0, 'lost_tp': 0, 'new_fp': 0,
               'removed_fp': 0, 'cases_with_lost_tp_or_new_fp': 0, 'flagged_cases': [],
               'before_tp_fp_fn': [0, 0, 0], 'after_tp_fp_fn': [0, 0, 0]}
    for key, spans in before['paired_mapped_spans']['gold'].items():
        gold = characters(spans)
        a = characters(before['paired_mapped_spans'][mode][key])
        b = characters(after['paired_mapped_spans'][mode][key])
        ca, cb = triple(gold, a), triple(gold, b)
        cases[key] = cb
        gu, au, bu = {offset for _, offset in gold}, {offset for _, offset in a}, {offset for _, offset in b}
        ua, ub = triple(gu, au), triple(gu, bu)
        paired_counts[mode][key] = {'typed_before': ca, 'typed_after': cb, 'untyped_before': ua, 'untyped_after': ub}
        uc = {'gained_tp': len((gu & bu) - au), 'lost_tp': len((gu & au) - bu),
              'new_fp': len((bu - gu) - au), 'removed_fp': len((au - gu) - bu)}
        for field, value in uc.items():
            untyped[field] += value
        for i in range(3):
            untyped['before_tp_fp_fn'][i] += ua[i]
            untyped['after_tp_fp_fn'][i] += ub[i]
        if uc['lost_tp'] or uc['new_fp']:
            untyped['cases_with_lost_tp_or_new_fp'] += 1
            untyped['flagged_cases'].append({'case_id': key, 'before_tp_fp_fn': ua, 'after_tp_fp_fn': ub, **uc})
        lost_tp, new_fp = (gold & a) - b, (b - gold) - a
        gained_tp, removed_fp = (gold & b) - a, (a - gold) - b
        changes = {'gained_tp': len(gained_tp), 'lost_tp': len(lost_tp),
                   'new_fp': len(new_fp), 'removed_fp': len(removed_fp)}
        for field, count in changes.items():
            aggregate[field] += count
        for kind in sorted({t for t, _ in gained_tp | lost_tp | new_fp | removed_fp}):
            counts = aggregate['by_type'].setdefault(kind, {field: 0 for field in changes})
            for field, positions in [('gained_tp', gained_tp), ('lost_tp', lost_tp), ('new_fp', new_fp), ('removed_fp', removed_fp)]:
                counts[field] += sum(t == kind for t, _ in positions)
        aggregate['changed_cases'] += a != b
        comp = 'lower' if sum(cb[1:]) < sum(ca[1:]) else 'higher' if sum(cb[1:]) > sum(ca[1:]) else 'equal'
        aggregate['character_errors_' + comp] += 1
        if lost_tp or new_fp:
            aggregate['cases_with_lost_tp_or_new_fp'] += 1
            aggregate['flagged_cases'].append({'case_id': key, 'before_tp_fp_fn': ca, 'after_tp_fp_fn': cb, **changes})
    aggregate['no_lost_tp_or_new_fp'] = aggregate['lost_tp'] == aggregate['new_fp'] == 0
    all_positions[mode] = aggregate
    untyped['no_lost_tp_or_new_fp'] = untyped['lost_tp'] == untyped['new_fp'] == 0
    for side in ('before', 'after'):
        tp, fp, fn = untyped[side + '_tp_fp_fn']
        untyped[side + '_metrics'] = {'precision': round(tp / (tp + fp), 6) if tp + fp else 0,
            'recall': round(tp / (tp + fn), 6) if tp + fn else 0,
            'f1': round(2 * tp / (2 * tp + fp + fn), 6) if 2 * tp + fp + fn else 0}
    untyped_positions[mode] = untyped
    closure[mode] = {}
    for key, old in prior['previous_regression_case_counts'][mode].items():
        a, b = old['before_regression'], cases[key]
        closure[mode][key] = {**old, 'now': b,
            'previous_character_count_regression_restored': b[0] >= a[0] and b[1] <= a[1] and b[2] <= a[2]}
report['all_aligned_mapped8_per_position'] = all_positions
report['previous_regression_case_restoration'] = closure
report['all_aligned_mapped8_untyped_per_position'] = untyped_positions
(ROOT / 'paired-all8-character-counts.json').write_text(json.dumps(paired_counts, indent=2) + '\n')
report['strict_regression_gate_passed'] = all(x['no_lost_tp_or_new_fp'] for group in (all_positions, untyped_positions) for x in group.values())
actual = {}
assert before['gold_alnum_offsets'] == after['gold_alnum_offsets']
for mode in modes:
    outcome = {'cases': before['aligned_rows'], 'gained_tp': 0, 'lost_tp': 0, 'new_fp': 0, 'removed_fp': 0,
               'cases_with_lost_tp_or_new_fp': 0, 'flagged_cases': [],
               'before_tp_fp_fn': [0, 0, 0], 'after_tp_fp_fn': [0, 0, 0]}
    for key, offsets in before['gold_alnum_offsets'].items():
        gold = set(offsets)
        a = set(before['actual_masked_alnum_offsets'][mode][key])
        b = set(after['actual_masked_alnum_offsets'][mode][key])
        ca, cb = triple(gold, a), triple(gold, b)
        paired_counts[mode][key].update({'actual_mask_before': ca, 'actual_mask_after': cb})
        changes = {'gained_tp': len((gold & b) - a), 'lost_tp': len((gold & a) - b),
                   'new_fp': len((b - gold) - a), 'removed_fp': len((a - gold) - b)}
        for field, count in changes.items():
            outcome[field] += count
        for i in range(3):
            outcome['before_tp_fp_fn'][i] += ca[i]
            outcome['after_tp_fp_fn'][i] += cb[i]
        if changes['lost_tp'] or changes['new_fp']:
            outcome['cases_with_lost_tp_or_new_fp'] += 1
            outcome['flagged_cases'].append({'case_id': key, 'before_tp_fp_fn': ca, 'after_tp_fp_fn': cb, **changes})
    outcome['no_lost_tp_or_new_fp'] = outcome['lost_tp'] == outcome['new_fp'] == 0
    actual[mode] = outcome
report['all_aligned_actual_mask_alnum'] = actual
report['strict_typed_gate_passed'] = all(x['no_lost_tp_or_new_fp'] for x in all_positions.values())
report['strict_untyped_gate_passed'] = all(x['no_lost_tp_or_new_fp'] for x in untyped_positions.values())
report['strict_actual_mask_gate_passed'] = all(x['no_lost_tp_or_new_fp'] for x in actual.values())
report['strict_regression_gate_passed'] = report['strict_regression_gate_passed'] and report['strict_actual_mask_gate_passed']
report['limitations'].append('Actual mask/restore executed for every row and mode; alphanumeric gold union uses the same eight mapped types. Raw output masking is evaluated without dropping any output candidate type. Off-scope gold types may contribute to absolute FP counts; the reported no-new-FP gate is paired between revisions.')
(ROOT / 'paired-all8-character-counts.json').write_text(json.dumps(paired_counts, indent=2) + '\n')
report['status'] = 'Frozen offline release validation; morphology-consistent policy with disclosed typed-category exceptions'
report['acceptance'] = {
    'criterion': 'Actual mask has no lost protected alphanumeric TP and no new masked non-gold alphanumeric FP, and aggregate F1 does not decrease in any reported scope/mode/metric; typed-category gate reported separately.',
    'actual_mask_gate_passed': report['strict_actual_mask_gate_passed'],
    'no_aggregate_f1_decrease': not any(x['field'] == 'f1' for x in report['observed_metric_regressions']),
    'typed_category_exceptions_disclosed': True,
}
report['acceptance']['passed'] = report['acceptance']['actual_mask_gate_passed'] and report['acceptance']['no_aggregate_f1_decrease']
report['residual_typed_category_cases'] = []
for row in all_positions['person_location']['flagged_cases']:
    key = row['case_id']
    report['residual_typed_category_cases'].append({**row,
        'actual_mask_before': paired_counts['person_location'][key]['actual_mask_before'],
        'actual_mask_after': paired_counts['person_location'][key]['actual_mask_after'],
        'interpretation': 'Generic identity-document series/number pair receives the established PASSPORT fallback; reference type is DRIVER_LICENSE. Morphological variants receive consistent masking; type remains ambiguous without an explicit owner.'})
key = prior['unresolved_category_diagnostic_case_id']
report['baseline_unresolved_category_case'] = {'case_id': key, **paired_counts['person_location'][key],
    'interpretation': 'The unqualified identity-document label with bare grouped digits does not establish a concrete passport or driving-licence category. Baseline FN remains; this is an unresolved ambiguity, not a new regression.'}
report['intermediate_trial_case_comparison'] = {key: paired_counts['person_location'][key]
    for key in prior['intermediate_trial']['flagged_case_ids']}
(ROOT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

main = all_positions['person_location']
main_untyped = untyped_positions['person_location']
lines = ['# RedMadRobot: проверка финальной версии', '',
    f"Код `{before['revision'][:7]}` → `{after['revision'][:7]}`. Строк предложено {before['offered_rows']}, выровнено {before['aligned_rows']}, прежних исключений {len(before['excluded_rows'])}. Все baseline-метрики точно совпали с предыдущим отчётом версии `{before['revision'][:7]}`.", '',
    'Главный estimand: common5, whole-case selection по исходным типам gold, typed-character micro-F1, PERSON+LOCATION. NER offsets одинаковы; score восстановлен как 0.85. Это повторное измерение ранее использованного корпуса.', '',
    '| Режим common5 | Character F1 до → после | Exact-span F1 до → после | Exact cases до → после |',
    '|---|---|---|---|']
for mode, values in report['scopes']['common5']['systems'].items():
    a, b = values['before'], values['after']
    lines.append(f"| {mode} | {a['typed_character']['f1']:.6f} → {b['typed_character']['f1']:.6f} | {a['merged_exact_span']['f1']:.6f} → {b['merged_exact_span']['f1']:.6f} | {a['typed_character']['exact_cases']} → {b['typed_character']['exact_cases']} |")
lines += ['', '**Приёмка по actual-mask и отсутствию падения агрегатных F1: ' + ('PASS' if report['acceptance']['passed'] else 'FAIL') + '.**', '', '**Строгая проверка всех typed/untyped позиций: ' + ('PASS' if report['strict_regression_gate_passed'] else 'FAIL') + '.**', '',
    f"PERSON+LOCATION, все {main['cases']} строк и восемь сопоставимых типов: +{main['gained_tp']} TP, −{main['lost_tp']} прежних TP; +{main['new_fp']} новых FP, −{main['removed_fp']} прежних FP. В {main['cases_with_lost_tp_or_new_fp']} случаях есть потерянные TP или новые FP. Сумма FP+FN выросла в {main['character_errors_higher']} случаях; уменьшилась в {main['character_errors_lower']}.", '',
    f"Actual mask alphanumeric gate: {'PASS' if report['strict_actual_mask_gate_passed'] else 'FAIL'}; main gainedTP={actual['person_location']['gained_tp']}, lostTP={actual['person_location']['lost_tp']}, newFP={actual['person_location']['new_fp']}, removedFP={actual['person_location']['removed_fp']}.", '',
    f"Untyped coverage: +{main_untyped['gained_tp']} TP, −{main_untyped['lost_tp']} прежних TP; +{main_untyped['new_fp']} новых FP, −{main_untyped['removed_fp']} прежних FP. Случаев с потерями или новыми FP: {main_untyped['cases_with_lost_tp_or_new_fp']}.", '',
    '| Случай | TP/FP/FN до | TP/FP/FN после | Потерянные TP | Новые FP |', '|---|---|---|---:|---:|']
for row in main['flagged_cases']:
    lines.append(f"| {row['case_id']} | {row['before_tp_fp_fn']} | {row['after_tp_fp_fn']} | {row['lost_tp']} | {row['new_fp']} |")
restored = sum(x['previous_character_count_regression_restored'] for x in closure['person_location'].values())
lines += ['', f"Из {len(closure['person_location'])} случаев предыдущего diagnostic audit восстановлены {restored} по TP/FP/FN относительно состояния до предыдущей регрессии. Это не значит, что все эти строки стали идеальными: оставшиеся прежние ошибки сохранены в report.", '',
    '| Срез PERSON+LOCATION | Строк | Character F1 до → после | Exact-span F1 до → после |', '|---|---:|---|---|']
for name, scope in report['scopes'].items():
    v = scope['systems']['person_location']; a, b = v['before'], v['after']
    lines.append(f"| {name} | {scope['cases']} | {a['typed_character']['f1']:.6f} → {b['typed_character']['f1']:.6f} | {a['merged_exact_span']['f1']:.6f} → {b['merged_exact_span']['f1']:.6f} |")
lines += ['', f"Все ухудшившиеся агрегаты ({len(report['observed_metric_regressions'])}) перечислены в `observed_metric_regressions`. Рост common5 F1 не отменяет выявленные регрессии. Новая статистическая проверка значимости не проводилась.", '',
    'Три новых typed FP относятся к личным документам с общей формулировкой удостоверения: цифры теперь закрываются, но default PASSPORT не совпадает с DRIVER_LICENSE в разметке. Это заявленная остаточная ошибка категории, а не доказанное отсутствие всех регрессий. Ограничение по форме слова ради legacy regex не вводилось.', '',
    'Случай row_2256 сохраняет baseline-пропуск: конкретный тип документа из общего удостоверения личности и голых групп цифр не устанавливается. Поэтому результат не равен 100% качества.', '',
    'Числовые отчёты содержат типы, offsets и counts, без исходных текстов. Исходные corpus/cache и метки не изменялись; измерение и агрегация не выполняют модельную инференцию или сеть.', '']
(ROOT / 'summary.md').write_text('\n'.join(lines))
validation = {'all_input_hashes_equal': True, 'all_scope_id_hashes_equal': True,
    'baseline_exactly_reproduces_previous_after_scopes': True,
    'mappings_equal_between_versions': True, 'frozen_gold_spans_identical': True,
    'before_checks': before['validation'], 'after_checks': after['validation'],
    'runtime_python_before': before['python'], 'runtime_python_after': after['python'],
    'strict_regression_gate_passed': report['strict_regression_gate_passed'],
    'strict_typed_gate_passed': report['strict_typed_gate_passed'],
    'strict_untyped_gate_passed': report['strict_untyped_gate_passed'],
    'strict_actual_mask_gate_passed': report['strict_actual_mask_gate_passed'],
    'declared_acceptance_passed': report['acceptance']['passed'],
    'aggregate_uses_only_numeric_reports': True, 'new_bootstrap': False,
    'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/x for x in ('runner.py', 'aggregate.py', 'prior-reference.json', 'report.json', 'summary.md', 'paired-all8-character-counts.json')]}}
for name in ('baseline.json', 'current.json'):
    path = ROOT / name
    if path.exists():
        validation['artifact_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        validation['artifact_sha256'][name] = hashlib.sha256(gzip.decompress(path.with_suffix('.json.gz').read_bytes())).hexdigest()
(ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps({'strict_gate_passed': report['strict_regression_gate_passed'], 'main_lost_tp': main['lost_tp'], 'main_new_fp': main['new_fp'], 'main_flagged_cases': len(main['flagged_cases']), 'previous_cases_restored': restored}))

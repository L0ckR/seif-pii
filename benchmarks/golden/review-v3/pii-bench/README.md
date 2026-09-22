# Frozen PII-Bench replay: 4e46c86 → 1219108

PII-Bench проверка кандидата `12191083eea106edf85c7bbc6e60a140faf1b4c2` не обнаружила регрессий относительно `4e46c86a6f6c7bf91e9525968981ae47d7984aba`.

| Подмножество | Случаев | Rules exact-span F1 | Hybrid PERSON-only exact-span F1 |
|---|---:|---:|---:|
| domain/common4, основной | 612 / 900 | 0.992000 → 0.992000 | 0.973643 → 0.973643 |
| domain/supported8 | 787 / 900 | 0.882613 → 0.884581 | 0.865236 → 0.867181 |
| entity/common4 | 280 / 910 | 0.983666 → 0.983666 | 0.980736 → 0.980736 |
| entity/supported8 | 560 / 910 | 0.784785 → 0.794439 | 0.777132 → 0.786538 |

На всех 108 primary/secondary, type и domain срезах нет потерь TP и новых FP в exact-span и typed-character метриках. Во всём выровненном корпусе из 1810 примеров, отдельно для каждого профиля, добавлено 10 точных TP span, 110 typed TP символов и 98 фактически закрытых TP символов без учёта типа; FP не добавлены, прежние TP не потеряны. Domain улучшился в 2 примерах, entity — в 8. Common4 предсказания не менялись.

Все три паспортных пропуска trial 5719 (`passport_002`, `passport_015`, `passport_048`) восстановлены в обоих профилях: spans точно совпадают с прежним baseline 4e. Это проверено по сохранённым типам/смещениям, без просмотра текстов в accepted replay. `passport-recovery.json` хранит результаты этой проверки.

Baseline 4e предсказания воспроизвелись побайтово относительно прошлого отчёта; все восемь primary/secondary показателей совпали точно. Проверены SHA256 исходного корпуса, cache, протокола и всех модулей source tree относительно Git до и после запуска. Полные source-хеши — в `source-baseline.json` и `source-final.json`; хеши корпуса и cache — в `validation.json`; runtime — в `runtime.json`; хеши артефактов — в `artifact-sha256.json`.

Это development regression: общий фикс последовал в том числе за адресной диагностикой предыдущего PII-Bench trial. Набор не является новым слепым holdout. Исходные корпуса, cache, разметка и прежние отчёты не изменялись. При frozen replay детектор/пороги/модели не менялись, модель и сеть не вызывались.

Исторический протокол сохранён: hybrid использует одни и те же cached Presidio PERSON spans со score 0.85; LOCATION не добавляется. Runtime `/tmp/seif-presidio-313/bin/python`: Python 3.13.7, pyarrow 25.0.1, numpy 2.4.6; установленный presidio-analyzer 2.2.364 читается только как источник конфигурации постоянного score. Это не HTTP/RPS или PERSON+LOCATION измерение.

Запускались:

```bash
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/pii-bench-release/runner.py
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/pii-bench-release/case_audit.py
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/pii-bench-release/aligned_transitions.py
```

Для отдельного повторения inference выбирайте новый каталог:

```bash
/tmp/seif-presidio-313/bin/python output/generalized-fixes-v3/pii-bench-release/runner.py --output /tmp/seif-pii-bench-1219108-replay
```

Архивные scripts содержат исходные абсолютные пути. Нужны совпадающие с зафиксированными Git refs source trees и локальный корпус/cache; изменённый детектор отклоняется проверкой SHA. Audit scripts читают predictions из собственного каталога. Файлы пишутся с защитой от перезаписи.

`report-with-case-audit.json`, `summary-with-case-audit.json`, `validation-with-case-audit.json` объединяют первый frozen отчёт с попарным аудитом. `case-audit.json` хранит только агрегаты, `aligned-transitions.json` — абсолютные TP/FP/FN и дельты для всего корпуса, обоих splits и профилей. Исходные тексты в выходных артефактах отсутствуют.

В итоговом audit untyped service coverage проверяется на целых случаях/domain с полной исходной gold; per-type срезы используют exact/typed метрики. Верно закрытая сущность другого типа не считается ложным срабатыванием service coverage. Первый trial 5719 и промежуточный d6 сохранены отдельно, без перезаписи.

Дополнительная проверка: predictions 121 полностью совпали побайтово с промежуточным 822 на этом корпусе. `compact-summary.json` содержит краткий итог, `provenance.json` — source/dataset/runtime/runner/prediction hashes.

Перед заморозкой финальных артефактов перепроверены SHA256 всех файлов, перечисленных в manifests промежуточных trial 5719/d6/822: все остались неизменными.

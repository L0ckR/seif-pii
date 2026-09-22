# Triage SonarQube: snapshot 4c85827

Проанализирован отчёт SonarQube 26.9 / Python analyzer 5.31, профиль Sonar way (398 Python rules), все файлы как source. Исходный отчёт: 119 issues — 114 CODE_SMELL, 3 VULNERABILITY, 2 BUG; 47 CRITICAL, 49 MAJOR, 23 MINOR. Здесь приведён независимый разбор исходных строк, а не принятие метки анализатора за доказанную уязвимость.

## Пять замечаний вне CODE_SMELL

| Место snapshot | Правило / метка Sonar | Разбор | Приоритет |
|---|---|---|---|
| scripts/compare_presidio.py:239 | S2245 / VULNERABILITY | Контекстно ожидаемо: seeded Random(20260922) перемешивает только порядок benchmark-кейсов. | P3 |
| scripts/compare_presidio.py:242 | S2245 / VULNERABILITY | Контекстно ожидаемо: тем же RNG перемешиваются реализации при замере. Это не секрет и не nonce; CSPRNG здесь ухудшит воспроизводимость. | P3 |
| scripts/local_redis.py:274 | S1045 / BUG | **Подтверждён небольшой дефект:** JSONDecodeError наследует ValueError, поэтому попадает в предыдущий except и не получает предназначенное ему общее сообщение о повреждённом state. | P3 |
| seif/ner.py:136 | S7490 / BUG | Контекстно ожидаемо, вероятное ложное срабатывание: внутри timeout есть async with TaskGroup, его выход неявно await-ит задачи и даёт cancellation checkpoint. | P3 |
| web/app.js:58 | S2245 / VULNERABILITY | **Требует доказательства практического воздействия:** Math.random используется только fallback для payload_id. Обычная HTTPS-ветка использует crypto.randomUUID; restricted API отдельно проверяет tenant/API key. В anonymous demo ID участвует в выборе mapping, поэтому непредсказуемость не следует объявлять неважной. | P2 |

Для JSONDecodeError проведён изолированный probe: mock snapshot вызывает синтетическую ошибку JSON, main возвращает 1 и печатает parse-message вместо общего state-message. Реальные Redis-процессы не затрагивались. Исправление — поставить специфичный обработчик перед ValueError.

Для S7490: существующий тест ожидания capacity прошёл. Дополнительный mock HTTP probe с 50 000 символами и timeout 25 мс завершился за 25,84 мс с NerUnavailable; все четыре начатые задачи завершили finally. Это опровергает конкретное утверждение об отсутствии checkpoint на рабочем пути. [Python документирует неявное ожидание на выходе TaskGroup](https://docs.python.org/3/library/asyncio-task.html#task-groups).

Для UI достаточно улучшить fallback через crypto.getRandomValues либо явно обработать отсутствие Web Crypto. Успешная атака, утечка или обход API key не воспроизводились; такие утверждения по этому rule ID делать нельзя.

## Приоритетные production CRITICAL

В runtime Python (seif + scripts/ner_service.py) 27 CRITICAL: **20 S3776** про cognitive complexity, **5 S1192** про повторяющиеся строки, **2 S5754** про BaseException. Подтверждённых критических уязвимостей этот разбор не выявил.

Главные цели аккуратного рефакторинга:

1. seif/app.py:217 create_app: complexity 178. Фабрика содержит вложенные endpoints, lifespan, обработку vault/CPU/NER и ошибок. Выносить связные операции; не объединять это с изменениями детекции.
2. seif/detector.py:802 detect: complexity 124; затем _merge_ner_candidates:678 (42), _structured_addresses:531 (35). Проверять точное равенство предсказаний golden, не только общий F1.
3. seif/person_fields.py:153 _initial_candidates (46), scripts/ner_service.py:198 create_app (39), seif/structured_fields.py:232 _document_candidates (34), seif/person_fields.py:91 _joined_name (32), seif/ner.py:87 _chunk (31).
4. Повторяющиеся публичные error messages можно вынести в осмысленные constants. Повторение слова «паспорт» само по себе не означает баг детектора.

Оба S5754 — seif/app.py:395 и scripts/ner_service.py:271 — находятся в callback завершения CPU future. Исключение передаётся ожидающему coroutine через waiter.set_exception; при уже отменённом HTTP-запросе результат считывается, чтобы исключение с входными ПД не попало в event-loop logs. Это обоснованный контекст асинхронной передачи ошибки. Механическая замена на Exception или прямой reraise может ухудшить cancellation/privacy. Проверены существующие тесты отмены, удержания capacity и отсутствия PII в логах. Если объединять в helper, отдельно проверить BaseException для ещё активного waiter.

## Regex complexity не равно ReDoS

Есть четыре MAJOR S5843: два issue у seif/detector.py:70, по одному seif/person_fields.py:27 и seif/structured_fields.py:211. Это сложность чтения выражения и число вложений/ветвей; S3776 аналогично измеряет сложность управления функции. Они не доказывают экспоненциальный backtracking. Для такого вывода нужен конкретный неблагоприятный ввод и замеры роста либо иной доказательный анализ.

Повторно выполнены 33 существующих adversarial/custom-regex проверки: все прошли за 0,33 с. Они покрывают свои примеры, а не доказывают отсутствие любых возможных ReDoS. Дополнительно 5 тестов deadline/cancellation/redaction прошли за 0,70 с. Общая проверка: **38 passed**, исходники snapshot и рабочего репозитория не изменены.

Полный перечень 27 production CRITICAL с местами, rule IDs, классификацией, исходными SHA и результатами probe — в code-analysis-triage.json. Этот triage не эквивалентен внешнему скору хакатона и не устанавливает его формулу.

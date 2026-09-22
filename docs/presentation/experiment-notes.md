# Метрики для архитектурной презентации

Срез на 23 сентября 2026 года. Новые модели и нагрузочные тесты при подготовке
презентации не запускались. Все значения извлечены из сохранённых JSON-отчётов;
`experiment-metrics.json` содержит точный JSON pointer и SHA-256 каждого источника.

Дополнение после вопроса о декодере GLiNER: выполнен CPU replay неизменённых
предсказаний через текущие правила. Основной JSON уже обновлён: GLiNER
**97,4598 / 74,2888 / 70,4692%**, spaCy **96,8856 / 74,3979 / 65,3661%**.
Старые значения ниже сохранены как история, в JSON — `historical_mask_f1`.
Нового HTTP замера нет. Аудит **5 095 текстов / 5 179 окон** не обнаружил
RuBERT-подобного сдвига GLiNER. Подробности и ограничения:
[`gliner-recheck.md`](gliner-recheck.md).

## 1. История кандидатов: F1 фактического маскирования

Метрика считает защищённые буквенно-цифровые позиции исходного текста,
включает все исходные gold-категории и все предсказания. Она не требует правильного
типа или точной границы сущности. В таблице проценты.

| Кандидат | Organizer, 446 | PII-Bench, 1 810 | RedMadRobot, 2 839 | Область сравнения |
|---|---:|---:|---:|---|
| spaCy `ru_core_news_sm` 3.8.0 + правила | 96,8856 | 71,9545 | 59,4971 | A: прежний PERSON/LOCATION gateway |
| GLiNER 2.5 PyTorch + правила | 97,4598 | 71,7599 | 65,6560 | A: прежний PERSON/LOCATION gateway |
| GLiNER 2.5 ONNX + правила | 97,4598 | 71,7599 | 65,6560 | A: те же spans, что PyTorch |
| LFM2.5 350M official hybrid + правила | 94,5814 | 70,8583 | 53,6616 | A*: PERSON + address-proxy |
| RuBERT TRT, прежний сервис | 96,8791 | 74,0731 | 69,7101 | B: контроль до обновления |
| RuBERT TRT, текущий эксперимент | 97,1285 | 77,8639 | 93,9041 | B: word decoder, полный gateway и новые правила |

Это история сервисных конфигураций. Между A и B различаются правила/декодеры,
поэтому таблица не изолирует эффект одной модели. GLiNER остаётся выше текущего
RuBERT на organizer в данном историческом сравнении; универсального победителя
нет. Для осмысленного выбора важны все три корпуса и нагрузка.

GLiNER использовал выбранные ранее описания `described-names`, threshold=0.8,
FP32 CUDA. При ONNX-конверсии spans совпали на **всех 5 095 текстах**, различались
только малые значения float scores. Источник: `gliner_quality` →
`/parity_vs_native/{corpus}/identical_span_cases`.

У LFM нет обычной географической LOCATION. Для сервисного эксперимента
`identity.person_name` отображается в PERSON, а `contact.address` — в LOCATION
как явно обозначенный **address-proxy**. Другие 38 нативных категорий модели
в этой строке в NER gateway не передаются.

Отдельное применение полной LFM без правил СЕЙФ, все 40 категорий:

| Декодер LFM | Organizer | PII-Bench | RedMadRobot |
|---|---:|---:|---:|
| Native raw | 59,8633 | 77,2176 | 60,4366 |
| Official hybrid | 60,6800 | 84,1837 | 57,0922 |

**84,18% PII-Bench нельзя подставлять в сервисную строку LFM**: это иной способ
применения и иной набор выходных типов. Official hybrid в основном выигрывает
на TOKEN в PII-Bench, но ухудшает RedMadRobot. Подробная атрибуция находится в
соседней ветке `benchmarks/ner-models/lfm25-pii/error-attribution.json`.

## 2. Что дало обновление RuBERT

Фиксированный контроль — commit `0ab6d876008af27ee015167ea50488be9d488fb7`.
Новая конфигурация использует word decoder с исходными Unicode offsets,
исправляет пропущенные tokenizer-ом control/PDF-символы, передаёт все 21 нативные
категории через 14 типов gateway и объединяет их с обновлёнными правилами.

Прирост F1 маскирования: organizer **+0,2494 п.п.**, PII-Bench **+3,7908 п.п.**,
RedMadRobot **+24,1940 п.п.**. На уверенной части organizer, 329 случаев,
F1 маскирования текущего сервиса составляет **99,3156%**.

Точная исходная граница — отдельная метрика. После сохранения атомарных
компонентов имён/адресов exact-span F1 organizer снизился **89,5616→55,8140%**,
PII-Bench **67,0761→45,6751%**, RedMadRobot вырос **42,7411→86,6113%**.
Это существенная разница представления: первые два gold-корпуса часто выделяют
ФИО целиком, RMR — отдельные части. Маскирование и точное восстановление
оцениваются независимо. Улучшение маски нельзя называть улучшением всех метрик.

## 3. Сравнение с pii-guard

| Метрика | СЕЙФ RuBERT после обновления | pii-guard reference |
|---|---:|---:|
| Organizer: full-mask F1 | 97,1285% | 81,8104% |
| PII-Bench: full-mask F1 | 77,8639% | 78,5479% |
| RedMadRobot: full-mask F1 | 93,9041% | 93,6748% |
| RMR: exact-span F1, common14 | 86,7601% | 89,8458% |
| RMR: full-mask precision | 97,2250% | 94,6488% |
| RMR: full-mask recall | 90,8026% | 92,7205% |

На RMR СЕЙФ имеет немного больший mask F1 за счёт precision, но меньший recall.
**Полного превосходства над pii-guard не достигли.** До правил наш исправленный
word decoder даёт на RMR exact common14 **89,5165%** и full-mask F1 **94,1150%**;
это модель отдельно, а не качество сервиса целиком.

Reference фиксирует upstream `24230abb72949a9f85499244dd4f15a0ad0cdd9e`: правила,
email gate, word-decoded NER и merge на исходном тексте. Не выполнялись
нормализация, транслитерация, преобразование английских числительных и base64.
Таким образом, reference — **raw-text ablation, не полный pipeline** и не
воспроизведение опубликованного значения 88,9.

Common14 ограничивает предсказания 14 семействами RMR одинаково для всех систем.
Исключены 19 предсказаний СЕЙФ и 253 guard; в основном guard DATE_TIME, которого
нет в таксономии RMR. При сохранении всех предсказаний exact F1 составляет
соответственно **86,6113% / 87,8188%**. Full-mask метрики ничего из этого не
отбрасывают.

## 4. Скорость: разделять протоколы

| Протокол | Система | Успешных HTTP RPS | p95, мс | Запросов | Конкуренция |
|---|---|---:|---:|---:|---:|
| GLiNER, короткий | PyTorch FP32 CUDA | 31,89 | 176,04 | 446 | 4 |
| GLiNER, короткий | ONNX CUDA | 49,86 | 116,84 | 446 | 4 |
| RuBERT, контроль | Прежний сервис TRT | 528,86 | 20,01 | 22 300 | 8 |
| RuBERT, контроль | Текущий сервис TRT | 533,74 | 19,58 | 22 300 | 8 |

GLiNER и RuBERT следует показать отдельными группами: неодинаковы длина
прогона, конкуренция, версии окружений и конфигурации сервиса. Четыре строки
нельзя выдавать за единый контролируемый нагрузочный sweep.

Каждый HTTP запрос выполнил реальный NER. Ошибок HTTP, расхождений масок и типов
с зафиксированным reference нет. Кэш предсказаний не обслуживал запросы.
Тесты локальные, с памятью вместо Redis и выключенным capture, на RTX 4070 Ti SUPER.
Это не измерение Cloudflare, максимальной ёмкости кластера или production SLA.

Для двух RuBERT прогонов одинаковы interpreters, GPU, веса и workload —
50 полных циклов 446 текстов organizer, новые payload IDs. API: один worker
Python 3.14.7t, GIL disabled. NER: один worker Python 3.12.11, TensorRT CUDA graphs,
четыре Torch CPU threads. Прогрев и restore-checks вне таймера RPS. Остановка NER
проверена: сервис отвечает 503. Наблюдение +0,92% — не доказанный статистический
выигрыш скорости; один before/after запуск не измеряет межзапусковую вариативность.

**LFM HTTP не измерялся**. Число **61,05 текста/с** относится к последовательным
GPU model calls с двумя декодерами на organizer. spaCy **343,89 текста/с** —
последовательный CPU model inference с одним потоком. Эти значения нельзя
помещать на шкалу HTTP RPS рядом с четырьмя строками выше.

## Как подключать JSON и проверять источники

В `experiments` первые шесть записей — кандидаты, седьмая — reference guard.
Число для графика качества: `experiments[i].mask_f1[corpus].value` (уже проценты).
HTTP: `.http.successful_mask_rps.value`, `.http.p95_ms.value`,
`.http.mask_requests.value`, `.http.concurrency.value`.
Для LFM/spaCy `.http` равен `null`: это отсутствие сопоставимого замера, не 0 RPS.
Каждая метрика содержит `.source` и `.json_pointer`; `sources[source]`
указывает рабочую копию, файл отчёта и его SHA-256.

Основные файлы:

- `gliner_quality`: `benchmarks/ner-models/gliner25-onnx/comparison-gpu.json`.
  Organizer: `/corpora/organizer/service_hybrid/{native|spacy|candidate}/unique_case_primary/character_metrics/f1`.
  Публичные корпуса: `/corpora/{pii|redmadrobot}/full_masking_all_gold_types/systems/{native|spacy|candidate}_hybrid/f1`.
- `lfm_quality`: в соседней рабочей копии `/home/lockr/projects/seif-pii-lfm/benchmarks/ner-models/lfm25-pii/comparison.json`.
  Organizer: `/corpora/organizer/gateway_profiles/lfm_hybrid_gateway_address_proxy/unique_case_primary/character_metrics/f1`.
  Публичные корпуса: `/corpora/{pii|redmadrobot}/gateway_profiles/full_masking_all_gold_types/systems/lfm_hybrid_gateway_address_proxy/f1`.
- `rubert_quality`: `benchmarks/ner-models/rubert-upgrade/quality-v5.json`.
  `/corpora/{corpus}/full_masking_all_gold_types/systems/{previous_service|upgraded_service|pii_guard_raw_word}/f1`.
- GLiNER HTTP: `benchmarks/ner-models/gliner25-onnx/http-cuda-v2.json`,
  `/backends/{0|1}/phases/1/successful_mask_rps`.
- RuBERT HTTP: `benchmarks/ner-models/rubert-upgrade/http-{baseline-control|upgraded-final}.json`,
  `/service/phases/0/successful_mask_rps`.

У всех корпусов сохранены прежние состав, тексты и разметка. Organizer —
предварительная AI silver-разметка, а не ground truth организаторов. Все три
корпуса уже изучались при разработке; независимого blind holdout здесь нет.
В RedMadRobot сохранены прежние два исключения выравнивания, новых исключений нет.

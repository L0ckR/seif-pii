# GLiNER 2.5 multilingual: воспроизводимый эксперимент

Дата: 22 сентября 2026. База: `23329a03a39c58617e80b819aea5149c86a317a7`.
Ветка: `experiment/gliner25-multi`. Production и его модель не переключались.

Проверяется замена только PERSON/LOCATION NER: `ru_core_news_sm` через Presidio
на `fastino/gliner2.5-multi-v1`. Правила российских/СНГ документов, объединение
кандидатов, фильтрация публичных сущностей и маскирование одинаковы. Обучения
весов, изменения эталонных меток и исключения ошибок после inference нет.

## Organizer: подбор конфигурации

446 уникальных текстов, SHA-256
`3bec44d5eafbffc799cd525bbe573f8b0adf84ec82e16271bd406fe92b521e46`.
Разметка — прежний зафиксированный AI silver, а не официальный ground truth
организаторов: 117 неоднозначных и 329 уверенных примеров, 77 отрицательных.
Это уже просмотренный development set. Оценка не равна баллу системы хакатона.

Главная метрика ниже — micro-F1 маскированных буквенно-цифровых символов по
существующему `scripts/evaluate_golden.py`; точность типа сущности не требуется.
Exact — полное совпадение защищённых символов примера, из 446.

| Конфигурация | Precision | Recall | F1 | Exact |
|---|---:|---:|---:|---:|
| spaCy, прежний сервис | 98,0072% | 95,7893% | 96,8856% | 415 |
| GLiNER `person-location`, 0.5 | 90,4409% | 97,4031% | 93,7930% | 387 |
| GLiNER `person-location`, 0.8 | 94,2290% | 96,0119% | 95,1121% | 400 |
| GLiNER `person-location`, 0.95 | 97,6994% | 94,5279% | 96,0875% | 414 |
| GLiNER `presidio-labels`, 0.5 | 89,9727% | 97,8668% | 93,7539% | 387 |
| GLiNER `described-names`, 0.5 | 97,1975% | 97,1434% | 97,1704% | 421 |
| **GLiNER `described-names`, 0.8** | **97,7782%** | **97,1434%** | **97,4598%** | **423** |
| GLiNER `described-names`, 0.95 | 97,7619% | 96,4200% | 97,0863% | 421 |

Первый запуск — простой перенос PERSON/LOCATION с порогом 0.5. Остальные
конфигурации исследованы после него на том же наборе, поэтому лучший результат
не является независимым подтверждением обобщения. Повтор первичной конфигурации
на CPU дал те же предсказания. Свежий spaCy дал полное совпадение всех 446 строк
и 221 сущности с прежним замороженным кэшем.

`described-names` задаёт описания имён людей и географических названий,
исключающие роли, названия полей и числовые идентификаторы. Дополнительная метка
`organization` помогает разделять типы, но не возвращается как PII. Это не
безусловный запрет пересечения с организацией. Точные описания находятся в
[`seif/gliner_ner.py`](../../../seif/gliner_ner.py) и metadata каждого запуска.
Порог, описания и FP32 зафиксированы до оценки новых внешних корпусов.

Окончательное подтверждение выбранного варианта прошло через тот же
`GlinerAnalyzer` и `scripts.ner_service.infer`, которые использует сервис:
[`selected-service.jsonl`](selected-service.jsonl) побайтно совпадает с
`gpu-described-names-threshold08.jsonl`. Проверены реальные Unicode offsets,
содержимое подстрок и числовые confidence. Длинный ввод обрабатывается окнами
384 слова с перекрытием 64; короткий — с `max_len=385`, учитывающим добавляемую
upstream точку. Поле `max_len: null` в metadata этого запуска относится к
прямому пути/warmup; `inference_path: service-adapter` и исходный код фиксируют
описанный предел сервисного пути.

### Что означает улучшение

Выбранный вариант добавляет 73 правильно защищённых символа и 14 лишних,
уменьшая FN с 227 до 154. Десять масок стали лучше, две хуже. Все 12 изменений
находятся среди 117 неоднозначно размеченных примеров. На уверенных 329
характеристики **одинаковы**: F1 99,4859%, FP 42, FN 0, 328 точных масок.

Девять улучшений возвращают пропущенные имена (54 символа); `org_0381`
закрывает ещё 20 символов паспортов, но ошибочно типизирует их как PERSON.
`org_0207` теряет один символ инициала, `org_0249` маскирует похожее на фамилию
слово из неоднозначно размеченного отрицательного примера (+14 FP).
Отрицательных примеров с лишней маской стало 7 вместо 6.

Вторичные типизированные метрики показывают компромисс: character F1
93,1895% → 93,4214%, **exact typed-span F1 91,7927% → 91,0828%**.
Заявлять улучшение всех метрик извлечения или гарантировать рост официального
score по этим данным нельзя. Полные агрегаты, изменения по case_id и контроль
правил: [`comparison-selected-service.json`](comparison-selected-service.json).

## Внешние корпуса

Новые сравнения используют HiveTrace PII-Bench (900 domain + 910 entity)
и Red Mad Robot (2839 из 2841 после прежних двух исключений BIO-выравнивания).
Прежние версии, словари типов и правила выбора срезов сохраняются; исходные
тексты находятся вне Git в игнорируемых каталогах. Порог 0.8 и описания не
подбираются по этим результатам. Корпуса ранее влияли на разработку правил,
поэтому это проверка переноса новой NER-конфигурации, не полностью слепой тест
всей системы. Наличие публичного корпуса также не доказывает его отсутствие
в неизвестных обучающих данных pretrained-модели.

[`corpus-overlap.json`](corpus-overlap.json) подтверждает отсутствие пересечений
трёх корпусов по исходному тексту и после casefold/нормализации пробелов.
Внутри публичных наборов повторов также нет. Organizer содержит 411 уникальных
нормализованных текстов из 446; исходный протокол и его 446 строк сохранены.

## Время и среда

Один последовательный прогретый вызов модели на текст, batch=1. Это **не
HTTP RPS**, нагрузочный предел сервиса или измерение с Redis. На organizer:

| Модель и устройство | Текстов/с | p50, мс | p95, мс |
|---|---:|---:|---:|
| spaCy, CPU, 1 BLAS thread | 343,89 | 2,76 | 4,20 |
| GLiNER primary 0.5, GPU FP32 | 31,27 | 24,76 | 63,10 |
| GLiNER primary 0.5, CPU, 4 threads | 8,54 | 115,17 | 139,07 |
| GLiNER selected 0.8, GPU, service adapter | 29,76 | 27,56 | 61,08 |

GPU: RTX 4070 Ti SUPER 16 GB. Peak allocated у выбранной модели около 1,27 GB
(не полное использование VRAM процессом). GLiNER существенно тяжелее spaCy;
данные не дают основания переносить прежние показатели RPS на этот backend.
FP16, компиляция, batching и масштабирование здесь не измерялись.

Обычный CPython 3.13.7 для NER, `gliner2==2.0.0`, `torch==2.14.0+cu130`,
`transformers==4.57.6`, FP32, compile=False. API отдельно остаётся на CPython
3.14t. Полная фиксация среды: [`requirements-gliner.txt`](../../../deploy/ner/requirements-gliner.txt).
Метаданные результатов содержат SHA-256 модели, данных, исходников, версии,
устройство и длительности каждого вызова. Первичные исследования сохранены
с хешами версии runner на момент запуска; финальный `selected-service` записан
после фиксации общего parser и сервисного пути.

## Что взяли из примера Presidio

[Пример GLiNERRecognizer](https://presidio.dataprivacystack.org/samples/python/gliner/)
полезен сопоставлением `person`/`name` с PERSON и разделением организации и
географии. В нашем опыте замена NER также не складывает ответы двух моделей.
Метки из примера проверены отдельной абляцией `presidio-labels`.

Однако пример использует старый `gliner.GLiNER`; для [этой модели](https://huggingface.co/fastino/gliner2.5-multi-v1)
нужен `gliner2.AutoExtractor`, выбирающий BoundaryExtractor. Legacy-параметры
`multi_label` и `flat_ner` нельзя просто передать новой библиотеке. Наш режим
`overlap_policy="flat"` — не точное воспроизведение всех параметров примера.

Модель загружается только с локального пути. Snapshot revision
`a221b77a8baf4a613b8f8652661d41fa10a5641e` закреплён при загрузке, поскольку
upstream AutoExtractor не передаёт все hub kwargs concrete loader. Веса,
tokenizer и config не менялись. Установленные protobuf/sentencepiece позволяют
встроенному compatibility fallback обработать legacy `extra_special_tokens`;
SDPA недоступен для этого encoder, библиотека использует eager attention.

## Воспроизведение

Из корня репозитория, после установки обычного Python 3.13 и `uv`:

```bash
uv venv --python 3.13 .venv-gliner
uv pip install --python .venv-gliner/bin/python -r deploy/ner/requirements-gliner.txt
.venv-gliner/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "fastino/gliner2.5-multi-v1",
    revision="a221b77a8baf4a613b8f8652661d41fa10a5641e",
    allow_patterns=["config.json", "encoder_config/config.json", "model.safetensors",
                    "tokenizer.json", "tokenizer_config.json"],
)
print(path)
PY
```

Для нового измерения задайте `MODEL_PATH` напечатанным локальным путём. Выход
должен быть новым: существующие результаты runner не перезаписывает.

```bash
.venv-gliner/bin/python scripts/cache_gliner.py \
  --model-path "$MODEL_PATH" --schema described-names --threshold 0.8 \
  --device cuda --service-adapter --output local-data/repeat/selected.jsonl
.venv-gliner/bin/python scripts/compare_ner_models.py \
  --candidate-cache local-data/repeat/selected.jsonl \
  --output local-data/repeat/comparison.json
```

Нативный экспериментальный NER-сервис запускается со следующим окружением:

```bash
export SEIF_NER_BACKEND=gliner
export SEIF_GLINER_MODEL_PATH="$MODEL_PATH"
export SEIF_GLINER_DEVICE=cuda
export SEIF_GLINER_SCHEMA=described-names
export SEIF_GLINER_THRESHOLD=0.8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# SEIF_NER_TOKEN задаётся отдельным секретом, одинаковым с API.
.venv-gliner/bin/python -m scripts.ner_service --host 127.0.0.1 --port 8770 --workers 1
```

API направляется на этот процесс через `SEIF_NER_URL` и `SEIF_NER_TOKEN`.
Умолчания сервиса остаются Presidio; экспериментальные умолчания GLiNER —
`person-location`, 0.5, CPU, поэтому выбранную конфигурацию задавайте явно.
Существующие Docker/Compose/ZIP по умолчанию содержат Presidio runtime; этот
эксперимент проверяет нативный backend, не готовый GPU Docker deployment.

Проверки: Ruff и полный unit suite — **2828 passed, 5 skipped**; пропуски —
opt-in Redis/Sentinel integrations без выделенного тестового окружения.
Отдельные adapter-тесты проверяют смещения, длинные тексты, повреждённые ответы,
схемы, пороги и отсутствие незаявленных загрузок модели.

[`http-smoke.json`](http-smoke.json): **PASS**, настоящие локальные GLiNER NER
и API на отдельных временных портах. Проверены 401/422, корректные Unicode
границы, маскирование PERSON+EMAIL и точное восстановление для короткого ввода
и 457 слов с именем в конце. API: Python 3.14.7t, GIL выключен. Использовались
новые секреты, memory vault и синтетические строки; тестовые процессы завершены.
Это проверка нативного сервиса, не Docker и не нагрузочное тестирование.

# GLiNER2.5 ONNX: фиксированное сравнение с native PyTorch

Экспериментальная ветка `experiment/gliner25-onnx` от GLiNER `1fffea1`.
Проверяется [`DanKau/gliner2.5-multi-v1-onnx`](https://huggingface.co/DanKau/gliner2.5-multi-v1-onnx/tree/481ad683a5420349c20f7ccc992efd25f9f3b809)
revision `481ad683a5420349c20f7ccc992efd25f9f3b809`.
Production и существующие ZIP/контейнеры не переключаются на этот backend.

## HTTP RPS

**49,86 mask RPS у ONNX против 31,89 у PyTorch** при concurrency=4,
то есть наблюдаемый прирост **56,4%**. Это реальные `/v1/mask` запросы
через API + NER, не выдача заранее сохранённых предсказаний.
[`http-cuda-v2.json`](http-cuda-v2.json) — полный успешный отчёт.

| Backend | Одновременных запросов | Успешных mask RPS | p50, мс | p95, мс | p99, мс | max, мс |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch | 1 | 25,33 | 35,95 | 61,51 | 83,87 | 194,52 |
| ONNX | 1 | 35,35 | 24,46 | 57,76 | 71,95 | 100,88 |
| PyTorch | 4 | 31,89 | 119,47 | 176,04 | 213,30 | 224,02 |
| ONNX | 4 | 49,86 | 75,10 | 116,84 | 140,60 | 151,36 |

Каждая строка — все 446 organizer-текстов с новыми payload IDs; всего
**1784 измеренных mask-запроса**. Во всех четырёх фазах:

- 446 HTTP 200, 446 реально завершённых NER-вызовов, 0 ошибок;
- 446/446 совпадений масок, типизированных границ и списков типов с reference;
- 0 запросов дольше 500 мс;
- 8/8 проверок точного восстановления вне timed window;
- после остановки собственного NER новый mask и health возвращают 503;
  тихого продолжения без NER нет.

Один API worker на Python 3.14.7t с выключенным GIL, один NER worker на
Python 3.13.7, GPU RTX 4070 Ti SUPER. Исполнитель модели однопоточный,
NER допускает до четырёх принятых model jobs. In-memory vault, без Redis,
capture и текстового кэша; idempotency replay исключён свежими IDs.
Счётчики actual model calls и NER-stage сверяются после каждого пакета.

Это короткий **closed-loop** замер текущей конфигурации: следующая работа
отправляется после завершения предыдущей на данном клиенте. Он не подтверждает
1000/2000 RPS из критериев хакатона, устойчивость за пять минут, максимальную
пропускную способность GPU с batching или масштабирование реплик. HTTP rate
выше offline model rate возможен из-за отдельного запуска, прогрева и условий
измерения; корректное сравнение HTTP — между строками одного этого теста.

Первый HTTP-запуск [`http-cuda.json`](http-cuda.json) не завершил postflight:
benchmark-клиент переиспользовал служебное соединение после server keepalive
timeout. Его результаты не засчитываются. Клиент исправлен: GET открывает
свежее соединение, простаивавший POST connection заменяется **до** отправки;
отправленные timed POST не повторяются и ошибки не скрываются.

## Результат качества

На GPU **все 5095/5095 текстов обработаны без ошибок**, границы и типы всех
2983 NER-сущностей совпали с замороженными PyTorch-предсказаниями.
Максимальная разница confidence — 0,00000614; она не изменила отбор spans
или итоговое качество сервиса. [`comparison-gpu.json`](comparison-gpu.json)
содержит все агрегаты, данные о скорости и подтверждённые хеши.

| Корпус | Текстов | F1 маскирования PyTorch | F1 ONNX | Полностью совпавшие с gold маски, оба backend |
|---|---:|---:|---:|---:|
| Organizer | 446 | 97,4598% | 97,4598% | 423/446 |
| PII Bench | 1810 | 71,7599% | 71,7599% | 1143/1810 |
| Red Mad Robot | 2839 | 65,6560% | 65,6560% | 1094/2839 |

Это равенство двух реализаций одной модели, а не повышение качества или
идеальная разметка организаторов. Публичные exact-span/typed-character метрики
и PERSON-only sensitivity тоже сохраняются в отчёте.

## Скорость модели

Нативный analyzer, последовательные вызовы, FP32, batch=1, прогрев исключён.
Это **тексты/с, не HTTP RPS**. Для ONNX указаны 4 intra-op потока на сессию
(encoder и boundary — две отдельные сессии), Torch preprocessing — 4 потока.

| Замер | Текстов/с | p50, мс | p95, мс |
|---|---:|---:|---:|
| PyTorch GPU, organizer, прежний замер | 29,76 | 27,56 | 61,08 |
| ONNX GPU, organizer | 43,78 | 19,34 | 44,61 |
| ONNX GPU, PII Bench | 46,02 | 18,04 | 44,12 |
| ONNX GPU, Red Mad Robot | 43,36 | 17,93 | 51,61 |
| PyTorch CPU, organizer, свежая выбранная схема | 4,68 | 208,23 | 253,48 |
| ONNX CPU, organizer, та же выбранная схема | 8,40 | 115,61 | 145,95 |

На organizer наблюдаемое ускорение GPU относительно прежнего PyTorch-замера —
**1,47×**. Все 5095 ONNX GPU документов с сериализацией кэша — 44,24 текста/с.
Предыдущие 8,54 текста/с CPU относились к первичной простой схеме меток,
поэтому здесь заново измерена выбранная `described-names` при том же пороге 0.8.
Её CPU-предсказания тоже совпали с GPU на 446/446 текстах:
[`native-cpu-selected.meta.json`](native-cpu-selected.meta.json),
[`native-cpu-vs-gpu-parity.json`](native-cpu-vs-gpu-parity.json).
ONNX CPU обработал все 446 текстов без ошибок, тоже с идентичными spans;
его F1 organizer — те же 97,4598%. Ускорение относительно выбранной схемы
PyTorch CPU — **1,80×**. [`comparison-cpu.json`](comparison-cpu.json).
Это отдельные однопроходные замеры на том же оборудовании; распределение между
повторными запусками и предельный RPS кластера ими не установлены.

Загрузка и освобождение runtime не входят в модельные timings. После печати
CPU-результата наблюдалось отложенное завершение процесса; он завершился сам
с кодом 0, без traceback. Причина задержки не установлена, профиль shutdown
не снят. Эти измерения не подтверждают время старта/остановки реплики.

## Обнаруженное ограничение экспорта

[`wrapper-smoke-v2.json`](wrapper-smoke-v2.json) подтверждает равенство на
8 синтетических проверках: прогрев, имена, Unicode, literal special tokens,
длинный текст с сущностью в конце и пустой ввод. Профили ORT показывают
выполнение encoder и boundary на CUDA; часть shape/control ops остаётся CPU.

На девятом входе — строке **`!`** — native возвращает пустой список сущностей,
а исходный ONNX-граф падает в `/head/shared_pool_builder/Where_14`:
формы `[1,16]` и `[1,28]` не совмещаются. Это воспроизведённый дефект экспорта;
он не исправлен подменой входа или скрытым переходом на PyTorch.
Следовательно, успешные 5095 корпусных примеров **не означают готовность
этого экспорта к безусловной замене production NER**.

Отдельно исправлена ошибка нашего адаптера: число ONNX-кандидатов динамическое,
от 0 до 192, а не всегда 192. Первоначальный запуск остановился на прогреве,
до оценки корпуса. После исправления guard и добавления тестов заново
зафиксирован [`protocol-v2.json`](protocol-v2.json); первоначальный протокол
сохранён для аудита. Веса, порог, описания меток и разметка не менялись.
Runner сохраняет модельные ошибки и отказывается выдавать метрику качества
полного корпуса при хотя бы одном сбое; ошибочные тексты не исключаются молча.

## Проверки кода

Полный suite в сервисной среде Python 3.14t: **2876 passed, 27 skipped**.
Из пропусков 22 относятся к необязательным NumPy/Torch-проверкам ONNX backend
и отдельно выполняются в модельной среде; остальные 5 — opt-in Redis/Sentinel.
В Python 3.13 с реальными backend-зависимостями все три новых тестовых файла:
**63 passed**, включая все 37 adapter-тестов. Ruff и `git diff --check` проходят.
Два предупреждения Starlette/AnyIO о deprecated API не являются падениями.

Независимо сверены хеши исходников/артефактов, равенство quality-агрегатов,
HTTP-счётчики и пересчёт RPS/percentiles по всем сохранённым latency rows.

## Протокол

Те же 5095 текстов: organizer 446, HiveTrace PII Bench 1810,
Red Mad Robot 2839 с прежними двумя исключениями ошибок BIO-выравнивания.
Модель, данные, порядок, правила и оценщики фиксируются до inference.
Настройки из выбранного GLiNER: `described-names`, порог 0.8, FP32,
batch=1, 4 CPU threads, seed=20260922; TF32 отключён.
Успешный replay прежних baseline-агрегатов обязателен перед запуском.

Сравниваются полные корпуса и все исходные gold-типы. Маскирование считается
по буквенно-цифровым позициям; typed span/character дополнительно штрафуют
неверные классы и границы. Organizer содержит прежнюю предварительную AI-разметку,
а не официальный ground truth. Эти корпуса уже участвовали в разработке правил,
а схема GLiNER выбиралась на organizer; это не слепой тест всего решения.
ONNX-настройки по результатам качества не подбираются.

## Происхождение и адаптер

Экспорт указан как полученный из fastino revision `aaecfe45…`; наш reference —
`a221b77…`. Все пять native model/config/tokenizer-файлов этих revisions
эквивалентны по HF metadata. Отдельный [provenance.json](provenance.json)
содержит проверенные идентификаторы. ONNX tokenizer.json также совпадает.

`seif/gliner_onnx.py` заменяет encoder и boundary head на ONNX Runtime.
Остаются исходные GLiNER processor, schema descriptions, word/query pooling,
abstention, выбор непересекающихся spans и объединение длинных окон.
Это важно: упрощённая инструкция ONNX-карты не описывает все детали native
декодирования. Наше `flat` использует максимум суммы confidence интервалов,
а не жадный выбор одного максимального span.

Экспериментальный адаптер **по-прежнему требует Torch и native checkpoint**
для исходного runtime/препроцессинга. Нейросетевые encoder/boundary вычисления
идут через ORT; это не готовая поставка без PyTorch и не Java-реализация автора.
FP16, INT8, TensorRT и batching в этот фиксированный эксперимент не входят.

Зависимости: [`requirements-gliner-onnx.txt`](../../../deploy/ner/requirements-gliner-onnx.txt).
ONNX Runtime GPU 1.30.0, ONNX 1.23.0, GLiNER2 2.0.0, Torch 2.14.0+cu130,
Transformers 4.57.6, Python 3.13.7. GPU — RTX 4070 Ti SUPER.
Локальные файлы обязательны, автоматических загрузок в inference нет.

## Воспроизведение

Создать Python 3.13 среду и установить lock-файл выше. Скачать pinned snapshot
ONNX и прежний pinned `fastino/gliner2.5-multi-v1@a221b77a8baf4a613b8f8652661d41fa10a5641e`.
`PUBLIC_RUN_DIR` — frozen public run из [GLiNER эксперимента](../gliner25-multi-v1/README.md).
`ONNX_PATH` и `NATIVE_PATH` — локальные snapshot-каталоги, `RUN_DIR` — новый
каталог внутри игнорируемого `local-data/` этого checkout.

```bash
.venv/bin/python scripts/compare_onnx.py prepare \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH"
.venv/bin/python scripts/compare_onnx.py cache \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH" \
  --backend onnx --device cuda --scope all
.venv/bin/python scripts/compare_onnx.py evaluate \
  --run-dir "$RUN_DIR" --public-run-dir "$PUBLIC_RUN_DIR" \
  --model-path "$ONNX_PATH" --native-path "$NATIVE_PATH" \
  --backend onnx --device cuda --scope all --output "$RUN_DIR/comparison-gpu.json"
```

Для CPU используются `--device cpu --scope organizer`. Измерения разных
моделей/устройств выполняются последовательно, без конкурирующего inference.
Повторная запись выходных файлов запрещена. Исходные тексты и model weights
не попадают в новые Git-артефакты.

HTTP-замер (после окончания остальных модельных запусков):

```bash
/path/to/python3.14t benchmarks/ner-models/gliner25-onnx/http_benchmark.py \
  --backend both --device cuda --model-path "$NATIVE_PATH" --onnx-path "$ONNX_PATH" \
  --ner-python "$PWD/.venv/bin/python" --api-python /path/to/python3.14t \
  --output "$RUN_DIR/http-repeat.json"
```

Каждый backend последовательно запускается на собственных loopback-портах,
с временными ключами и хранилищем. Benchmark останавливает только созданные
им процессы. Сравнение масок с frozen NER cache используется исключительно
для проверки ответа; cache не подключается к обслуживанию запросов.

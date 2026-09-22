# Два независимых RuBERT NER и общий API

Этот отдельный Compose-профиль поднимает две копии NER и две копии API.
Он не меняет основной `compose.yaml`, действующий туннель или текущий сервис.
Профиль предназначен для измерения масштабирования; **2000 успешных маскирований/с
не гарантируются самим числом контейнеров**.

```text
127.0.0.1:8785 → HAProxy API → API-1 ─┬→ общий Redis (vault + rate limit)
                            API-2 ─┤
                                   └→ private HAProxy → NER-1 → GPU 0
                                                     → NER-2 → GPU 0 или 1
```

NER работает в Python 3.12 с `torch==2.8.0+cu128` и `tensorrt-cu12==10.13.3.9`.
API использует существующий образ Python 3.14t. Каждый NER-процесс имеет
собственный экземпляр модели, TensorRT contexts и CUDA streams. Внутри одного
экземпляра нельзя просто снять блокировку: его mutable buffers не допускают
одновременного inference. Контейнеры обеспечивают независимость этих состояний.
CUDA user-space libraries устанавливаются закреплёнными wheel-зависимостями
Torch; отдельная установка CUDA toolkit внутри образа не требуется.

## Запуск

Нужны Docker Engine с Compose V2, NVIDIA Container Toolkit и совместимый драйвер
на Linux x86-64. Здесь не установлен Docker Engine, поэтому собранность образа,
запуск контейнеров, NVIDIA passthrough и RPS этого Compose-профиля ещё не проверены.
Конфигурация проверяется отдельно от производительности.

В `validation.json` сохранён результат проверки Compose CLI 2.39.4 и
`haproxy -c` для обоих конфигов на HAProxy 3.2.23, с SHA256 файлов. При проверке
вне Docker имена `ner-1`/`api-1` ещё не существуют; соответствующие DNS notices
ожидаемы и не заменяют будущую проверку связи контейнеров.
Короткая проверка маршрутизации использует настоящий HAProxy, локальные HTTP
stubs и по 64 POST через один keep-alive TCP. Оба backend получают запросы,
backend-соединения переиспользуются, все ответы 200. Это проверка маршрутизации,
без модели, Redis и исходных текстов, а не нагрузочный результат.

Команды выполняются из корня репозитория:

```bash
cp -n deploy/rubert/env.template .env.rubert
chmod 600 .env.rubert
```

Заполните `.env.rubert`:

| Переменная | Значение |
|---|---|
| `SEIF_MASTER_KEY` | Base64 от 32 случайных байт; один ключ для обеих API-реплик |
| `SEIF_DEMO_API_KEY` | Случайный ключ клиента `demo`; несмотря на имя, demo mode выключен |
| `SEIF_NER_TOKEN` | Случайный bearer token для внутреннего NER |
| `SEIF_REDIS_PASSWORD` | Случайный пароль внутреннего Redis |
| `SEIF_RUBERT_MODEL_HOST_PATH` | Абсолютный путь к локальному проверенному checkpoint |

Ключ можно сгенерировать через
`python3 -c 'import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())'`,
остальные секреты — через `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.
Не заменяйте ключ работающего сервиса: существующий vault требует прежнего ключа.
Этот Compose создаёт отдельный Redis и не использует хранилище активного стенда.
Файл `.env.rubert` игнорируется Git и Docker build context.

Модель должна быть заранее подготовлена оператором; Compose и сервер её не скачивают.
Каталог монтируется read-only, создание отсутствующего каталога запрещено. Нужны
семь файлов из `PINNED_FILES` в `seif/rubert_ner.py`:
`pii_ner.py`, `trt_backend.py`, `model.fp16.engine`, `config.json`,
`runtime_config.json`, `tokenizer.json`, `tokenizer_config.json`.
Сервис сверяет SHA256 каждого из них перед загрузкой и прогревает модель до readiness.
У пользователя контейнера UID 10001 должны быть права читать файлы и проходить каталог.
Веса из `local-data/` исключены из build context и не входят в образ/Git.

```bash
docker compose --env-file .env.rubert -f deploy/rubert/compose.yaml config --quiet
docker compose --env-file .env.rubert -f deploy/rubert/compose.yaml build
docker compose --env-file .env.rubert -f deploy/rubert/compose.yaml up -d --wait --wait-timeout 300
curl --fail http://127.0.0.1:8785/health
docker compose --env-file .env.rubert -f deploy/rubert/compose.yaml ps
```

`config --quiet` проверяет конфигурацию без печати секретов. Только вход API
публикуется на loopback; NER, балансировщик NER и Redis доступны через internal
Docker networks. Health API проверяет также Redis и NER. HAProxy проверяет NER
с bearer token, поэтому несовпадающий токен исключает backend из балансировки.
Образы NER одинаковые: второй контейнер использует слои первого, а не второй build.

```bash
# Остановка именно этого отдельного профиля:
docker compose --env-file .env.rubert -f deploy/rubert/compose.yaml down
```

## Параллелизм и batching

| Настройка | По умолчанию | Что ограничивает |
|---|---:|---|
| `SEIF_API_WORKERS` | 1 на контейнер | Число API-процессов, всего два |
| `SEIF_CPU_WORKERS` | 4 на процесс | CPU executor API |
| `SEIF_RUBERT_CPU_THREADS` | 4 на NER | Число intra-op CPU threads Torch; допустимо 1–32 |
| `SEIF_NER_MAX_CONCURRENCY` | 4 на процесс API | Одновременно отправленные в NER запросы |
| `SEIF_NER_MAX_MODEL_JOBS` | 4 на NER | Принятые задания, включая очередь; тот же `maxconn` у HAProxy |
| `SEIF_NER_MAX_HTTP_INFLIGHT` | 16 на NER | Все HTTP-запросы, включая health |
| `SEIF_NER_BATCH_SIZE` | 1 | Максимальный размер микробатча; допустимы 1/2/4/8/16/32 |
| `SEIF_NER_BATCH_WAIT_MS` | 1.0 | Короткое ожидание для формирования микробатча |
| `SEIF_NER_MEMORY` | 4 GiB на NER | RAM контейнера; это **не** лимит VRAM |
| `SEIF_API_MEMORY` | 1 GiB на API-контейнер | Общий лимит RAM всех его процессов |
| `SEIF_API_PROXY_MAXCONN` | 64 на API-контейнер | Backend capacity внешнего HAProxy |
| `SEIF_NER_PROXY_MAXCONN` | 256 всего | Соединения private HAProxy, включая idle keep-alive |
| `SEIF_REDIS_MAXMEMORY` | 1 GiB (`1gb`) | Лимит Redis с политикой `noeviction` |
| `SEIF_REDIS_MEMORY` | 1200 MiB (`1200m`) | RAM контейнера Redis; нужен запас сверх maxmemory |

Начальные значения сохраняют одиночный inference. Для эксперимента с batch 8
нужно согласованно увеличить model jobs и API concurrency как минимум до 8,
а HTTP capacity оставить с запасом для probes, например 32. Менять параметры
следует после проверки качества и соответствия измеренному профилю, а не считать
большую очередь способом увеличить вычислительную мощность. Большая очередь без
дополнительного throughput увеличивает задержки.

Балансировщик выбирает `leastconn` для каждого HTTP-запроса, сохраняет keep-alive
через `http-reuse safe` и явно отключает `prefer-last-server`. Сессия клиента
не закрепляется за одной NER-репликой. Сам HAProxy не повторяет POST автоматически;
NER-клиент API имеет отдельный ограниченный retry только на 429. У NER по умолчанию
максимум 256 принятых соединений на балансировщике и 100 мс
ожидания в очереди; у API — 1024 соединения и 250 мс. Перегрузка заканчивается
429/503 и должна учитываться как ошибка в нагрузочном тесте. Эти ограничения
нельзя превращать в «успешный RPS», считая только быстрые отказы.

## Пример: восемь API workers и два NER

`env.tuned-8api.template` — готовый **кандидат для нагрузочного тестирования**.
Он меняет только настройки, сохраняя секреты из `.env.rubert`. Это два
API-контейнера по четыре процесса; балансировщик знает два адреса, а распределение
соединений между четырьмя Uvicorn workers внутри контейнера выполняет их общий
listening socket. Равномерную загрузку восьми процессов нужно проверять по CPU
и числу запросов, а не выводить только из `SEIF_API_WORKERS=4`.

```bash
docker compose --env-file .env.rubert --env-file deploy/rubert/env.tuned-8api.template \
  -f deploy/rubert/compose.yaml config --quiet
docker compose --env-file .env.rubert --env-file deploy/rubert/env.tuned-8api.template \
  -f deploy/rubert/compose.yaml up -d --build --wait --wait-timeout 300
```

Настройки кандидата: API CPU executor=1, Torch threads=1, NER batch=32,
batch wait=2 мс, NER concurrency=16 на каждый API-процесс, model jobs=128 и
HTTP capacity=256 на каждый NER. Decoder и классы остаются `word`/`native`.
Нельзя заменять Torch threads настройкой `OMP_NUM_THREADS`: runtime явно вызывает
`torch.set_num_threads(SEIF_RUBERT_CPU_THREADS)`.

Восемь API-процессов допускают до 128 одновременных model-запросов и до 160
соединений к NER с учётом запасных слотов HTTP-пулов. Поэтому private HAProxy
получает global maxconn=512; прежний лимит 128 стал бы самостоятельным ограничением
до достижения всей мощности NER. Backend maxconn=128 на NER согласован с model
jobs; probe не создаёт model job и использует запас HTTP capacity. Внешний HAProxy
допускает 192 backend-соединения на каждый API-контейнер. Это ограничения очередей,
а не доказательство RPS. Суммарный объём очередей не должен заменять контроль p95/p99.

Обе API-реплики используют один Redis, один master key, одинаковый client API key
и одинаковые policies. Нельзя запускать независимые memory vault для каждого
процесса: восстановление через другую реплику и общий rate limit тогда потеряются.
Один Redis и один GPU остаются общими ресурсами этого профиля. Для подтверждения
2000 RPS нужен открытый поток запросов через весь этот путь с общим Redis;
измерение только NER или короткий closed-loop пик таким подтверждением не является.

## Одна или две GPU

По умолчанию `SEIF_NER_GPU_1=0` и `SEIF_NER_GPU_2=0`: два процесса используют
одну GPU. Они могут перекрыть подготовку запросов и независимые запуски, но делят
GPU/VRAM/CPU и иногда работают медленнее одного хорошо настроенного процесса.
Это экспериментальная стартовая конфигурация, а не обещание удвоения throughput.

Для двух совместимых GPU задайте `SEIF_NER_GPU_2=1` либо UUID устройства из
`nvidia-smi`. В каждом контейнере единственная назначенная GPU видна локально как
device 0, который использует runtime. Docker GPU reservation предоставляет доступ,
но не делает выбранную GPU эксклюзивной.

**Checkpoint TensorRT привязан к оборудованию.** Текущий engine собран для
RTX 4070 Ti SUPER, compute capability 8.9, TensorRT 10.13.3.9. Его совместимость
с другим GPU/драйвером нельзя вывести только из наличия CUDA. Если нужен новый
engine, его необходимо собрать отдельно, обновить проверяемый fingerprint и
повторно измерить качество. Ослаблять SHA256-проверку ради запуска не следует.
Ограничения описаны в [NVIDIA TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

## Что необходимо доказать для 2000 RPS

1. Сравнить один и два NER-процесса на одной GPU при одинаковых правилах,
   decoder=`word`, profile=`native`, корпусе и количестве API workers.
2. Для batch 1/4/8/16/32 проверить качество на всех 446 organizer-текстах и отдельно
   на PII-Bench/RedMadRobot; быстрые короткие строки не заменяют длинные документы.
3. Подать **2000 новых маскирований/с** открытым потоком, минимум 5 минут после
   прогрева, с уникальными request IDs, без готового кэша ответов. Отдельно считать
   отправленные, принятые, успешные, timeout, 429/503, p50/p95/p99 и пропуски генератора.
4. Включить в этот замер настоящий Redis, API replicas, маршрутизацию и тот режим
   записи запросов, который будет использоваться при проверке. Нагрузочный клиент
   не должен отбирать те же CPU, на которых уже упирается сервер.
5. Отключить одну NER-реплику и отдельно измерить восстановление, падение доступной
   мощности и сохранность восстановления текста через другую API-реплику.

## Границы профиля

Это один Docker-host с одним Redis: не кластер высокой доступности. Redis хранит
зашифрованные соответствия и общие лимиты, но persistence выключена как в базовом
Compose. При его остановке токены для последующего восстановления теряются.
Смена API-реплики при живом Redis и общем master key поддерживается. Перед
эксплуатацией важны отдельно проверенные persistence, репликация/failover и TTL.

Лимит клиента `demo` в существующих policies — 2500 RPS, общий через Redis;
два API не удваивают этот лимит. `/metrics` каждого API отражает его процессы,
поэтому Prometheus должен собирать обе реплики напрямую внутри сети.
`scripts/serve.py` создаёт `PROMETHEUS_MULTIPROC_DIR` в локальном `/tmp` контейнера
**до** запуска workers; `/metrics` агрегирует их через MultiProcessCollector.
Не задавайте общий metrics directory для двух контейнеров и не собирайте только
адрес gateway: в последнем случае запросы будут попадать в разные агрегаты и
искажать counter/rate. Для scrape нужны `X-System-ID: demo` и соответствующий
`X-API-Key`; этот маршрут также расходует общий лимит клиента.

Readiness API проверяет общий Redis и NER; readiness NER разрешается после
загрузки/fingerprint-проверки и прогрева модели. Начальный `depends_on` требует
обе здоровые реплики, а после запуска HAProxy исключает backend при неуспешном
health check. Сам `/health` NER не выполняет inference: отказ GPU при живом
процессе может не изменить этот флаг. Для такого случая нужны отдельное наблюдение
за inference errors/GPU и проверенный механизм восстановления; автоматический
failover по одним текущим probes не гарантируется.
Один успешный `/health` контейнера с четырьмя workers не доказывает здоровье
каждого процесса: Uvicorn supervisor отвечает за их перезапуск, а деградацию
мощности дополнительно контролируют метрики, CPU и error rate.
Базовый лимит Redis — 1 GiB; tuned overlay задаёт 4 GiB (`SEIF_REDIS_MAXMEMORY=4gb`)
и 5 GiB RAM контейнера (`SEIF_REDIS_MEMORY=5g`), оставляя запас под накладные расходы.
Это начальный запас ёмкости, а не результат проверки установившегося режима за
полные 900 секунд TTL. Для устойчивой работы размер vault нужно
рассчитать как RPS × TTL × средний размер зашифрованной записи с учётом накладных
расходов Redis. При 2000 новых запросов/с и TTL 900 с это до 1,8 млн живых записей;
лимит памяти нельзя принимать достаточным без отдельного замера.
Сопоставляйте прирост `used_memory` с приростом числа живых ключей, затем проверяйте
`used_memory_rss`, fragmentation и память контейнера на протяжении полного TTL.
`maxmemory` и ограничение RAM контейнера задаются отдельно; при заполнении
`noeviction` отклоняет новые записи, сохраняя уже принятые до их TTL.
Запись исходных запросов здесь не включена; её стоимость необходимо включать
в нагрузочный профиль отдельно, когда запись нужна.

Для нескольких хостов потребуется приватная сеть между API/NER, TLS/mTLS между
хостами, общий защищённый Redis и доступ к одинаковому проверенному артефакту.
Для Kubernetes GPU нужны NVIDIA device plugin, GPU resource requests, размещение
реплик на GPU-узлах, readiness и корректная остановка. CPU HPA сам по себе не
добавляет GPU. Существующая папка `deploy/k8s/` описывает CPU-профиль; этот Compose
не означает, что GPU Kubernetes-кластер уже развёрнут или проверен.

Конфигурация GPU соответствует [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/),
балансировка и очереди — [HAProxy 3.2 configuration manual](https://docs.haproxy.org/3.2/configuration.html).

# СЕЙФ — только сервис

API защиты и восстановления персональных данных, опциональный русскоязычный
NER-сервис, зависимости и конфигурация запуска.
Из дистрибутива удалён демо-интерфейс; API-маршруты и алгоритмы сохранены.

## Запуск с Redis и русскоязычным NER

Нужны Docker с Linux-контейнерами и Docker Compose. При первой сборке нужен
доступ в интернет для установки зависимостей и русской модели. Из каталога распакованного архива создайте
новый `.env` (существующий файл эта команда не перезаписывает):

```bash
python3 - <<'PY'
import base64, os, secrets
values = {
    "SEIF_MASTER_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
    "SEIF_DEMO_API_KEY": secrets.token_urlsafe(32),
    "SEIF_NER_TOKEN": secrets.token_urlsafe(32),
    "SEIF_DEMO": "0",
}
with os.fdopen(os.open(".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
    f.write("".join(f"{name}={value}\n" for name, value in values.items()))
PY
docker compose -f compose.yaml -f compose.ner.yaml up --build -d
```

API: `http://127.0.0.1:8765`. Проверка готовности: `GET /health`.
Русская модель устанавливается при сборке NER-образа из зафиксированного
`deploy/ner/requirements-ner.txt`; при старте сервис модель не скачивает.
Запуск только правил без NER: `docker compose up --build -d`.
По умолчанию NER использует четыре worker и лимит 4 CPU. На меньшем хосте
добавьте в `.env` `SEIF_NER_WORKERS=1` и `SEIF_NER_CPUS=1.0` перед запуском.
Это уменьшит требования к ресурсам и пропускную способность NER.

В режиме `SEIF_DEMO=0` передавайте `X-System-ID: demo` и `X-API-Key`
со значением `SEIF_DEMO_API_KEY` из созданного `.env`. Для проверки хакатона
без этих заголовков установите в `.env` `SEIF_DEMO=1` и оставьте
`SEIF_DEMO_API_KEY=` пустым перед запуском. Сохраните созданные
`SEIF_MASTER_KEY` и `SEIF_NER_TOKEN`.
Порт API в Compose опубликован только на localhost хоста; туннель направляется на порт 8765.

`POST /process` принимает:

```json
{"payload":"Email: user@example.com","payload_id":"unique-request-id"}
```

Ответ содержит `result`. Для восстановления передайте полученный `result`
в `payload` с тем же `payload_id` до истечения TTL. Контракт — `process_api.yaml`.
Также доступны `/v1/mask`, `/v1/unmask`, `/v1/types` и `/metrics`.
На `/` нет демо-страницы: ответ 404 для этого дистрибутива ожидаем.

## Локальный запуск API без контейнеров

Для режима правил достаточно Python 3.12+. Для работы без GIL используйте
Python 3.14.7t; основной Dockerfile уже настраивает эту версию.

```bash
python3.14t -m venv .venv
.venv/bin/pip install -r requirements.lock
SEIF_DEMO=1 SEIF_DEMO_API_KEY= SEIF_WORKERS=1 .venv/bin/python scripts/serve.py
```

Это однопроцессный запуск с хранилищем в памяти. Несколько workers требуют
общего Redis/Sentinel и одинакового стабильного `SEIF_MASTER_KEY`;
Compose настраивает общий Redis автоматически. Политики систем находятся
в `config/policies.yaml`. Захват текстов запросов по умолчанию выключен.

Целостность файлов после распаковки: `sha256sum -c SHA256SUMS`.

В архиве нет тестов, инструментов оценки и нагрузки, отчётов, датасетов,
CI, Kubernetes-обвязки, Git, локальных окружений и действующих секретов.

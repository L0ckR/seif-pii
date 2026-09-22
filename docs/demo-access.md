# Доступ к демонстрации

На момент подготовки работают два HTTPS-адреса:

| Профиль | Панель | Эндпоинт организатора |
|---|---|---|
| Правила РФ/СНГ, быстрый | https://left-rivers-emotions-doctor.trycloudflare.com | `POST https://left-rivers-emotions-doctor.trycloudflare.com/process` |
| Правила + Presidio PERSON/LOCATION | https://alpha-facility-muze-harmony.trycloudflare.com | `POST https://alpha-facility-muze-harmony.trycloudflare.com/process` |

Это Cloudflare Quick Tunnels к локальному стенду: они действуют, пока запущены компьютер, сервисы и процессы туннелей. После пересоздания туннеля адрес изменится. Публичная демонстрация предназначена для вымышленных данных; обычная конфигурация вне demo требует ключ системы.

Оба профиля используют один зашифрованный Redis/Sentinel vault. Для независимого сравнения разных профилей создавайте разные `payload_id`: повтор ID намеренно возвращает сохранённый результат первого профиля, сохраняя контракт ретраев.

По `/health` можно проверить готовность и `detector_profile`; `rules` означает профиль правил, `hybrid` — дополнительную локальную модель. Токен внутреннего NER-сервиса не публикуется. Прямой NER-порт через туннель не доступен.

Длительные HTTP-замеры выполнены локально, а не через Cloudflare; их нельзя переносить на внешний канал. [Измерения](validation.md), [инструкция жюри](judge-demo.md), [публикация и Kubernetes](kubernetes.md).

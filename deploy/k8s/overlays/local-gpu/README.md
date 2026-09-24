# Локальный GPU-профиль kind

Этот overlay запускает текущий API в трёх pod, RuBERT на GPU в двух pod и
Redis/Sentinel в трёх pod. Все они работают в kind-кластере `seif`. Модель
монтируется только для чтения через PVC, а захваченные запросы пишутся в
отдельный PVC. Домашний каталог WSL не монтируется в pod.

На WSL2 должны работать Docker, NVIDIA Container Toolkit, `nvidia-ctk`,
`python3` с PyYAML, `kind` и `kubectl`. Кластер `seif` должен существовать.
Укажите локальный каталог проверенного checkpoint RuBERT:

```bash
docker build -t seif-api:local-be55982 .
docker build -t seif-ner:rubert-be55982 -f deploy/rubert/Dockerfile .
bash deploy/k8s/overlays/local-gpu/setup-kind-node.sh /path/to/rubert/model
kind load docker-image seif-api:local-be55982 seif-ner:rubert-be55982 --name seif
kubectl apply -f deploy/k8s/base/namespace.yaml
python3 deploy/k8s/create-secrets.py  # Только для нового кластера без Secret.
kubectl apply -k deploy/k8s/overlays/local-gpu
kubectl -n seif rollout status deployment/seif-ner --timeout=10m
kubectl -n seif rollout status deployment/seif-api --timeout=10m
```

При переносе уже работающего сервиса сохраните прежний `SEIF_MASTER_KEY` в
`seif-secrets` до переключения трафика: иначе ранее выданные токены не
расшифруются. `ner-key` должен совпадать у API и NER. Секреты не кладите в
команды оболочки, логи, манифесты или Git.

Проверка внутри kind: `docker exec seif-control-plane curl -fsS
http://127.0.0.1:30066/health`. Чтобы опубликовать сервис на loopback WSL,
запустите прокси в Docker-сети kind:

```bash
docker run -d --name seif-kind-gateway --restart unless-stopped --network kind \
  -p 127.0.0.1:8767:8767 --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true \
  -v "$PWD/deploy/k8s/overlays/local-gpu/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro" \
  haproxy:3.2-alpine
curl -fsS http://127.0.0.1:8767/health
```

Публичная VM передаёт запросы на этот loopback-порт через постоянный обратный
SSH-туннель. На машине с прозрачным VPN обычный SSH уходил через медленный
VPN-шлюз; `scripts/ssh_bind_interface_proxy.py` фиксирует исходящее соединение
на физическом интерфейсе, сохраняя шифрование SSH. Для `ProxyCommand` укажите
`/usr/bin/python3 /path/to/seif-pii/scripts/ssh_bind_interface_proxy.py eth1 %h %p`
в пользовательском systemd-сервисе туннеля; замените `eth1` на нужный интерфейс.
Если прямой путь недоступен, скрипт пробует обычный маршрут. Проверяйте оба
звена после изменения: `http://127.0.0.1:8767/health` локально и
`https://seif.176-108-251-123.sslip.io/health` снаружи.

Второй, независимый публичный вход — именованный Cloudflare Tunnel на
`https://muravyinaya-ferma.online` и `https://api.muravyinaya-ferma.online`.
Он направляет HTTPS через Cloudflare сразу на `http://127.0.0.1:8767` и не
зависит от VM, Caddy или обратного SSH. Транспорт туннеля — QUIC; в текущей
сетевой конфигурации он идёт через VPN-маршрут WSL. Прямые маршруты к Cloudflare
edge, использовавшиеся при отладке HTTP/2, отключены.
Локальный конфиг и учётные файлы находятся в `~/.cloudflared/`, вне Git;
пользовательский `seif-cloudflare-tunnel.service` включён для автозапуска.
Прежний адрес через VM остаётся доступен параллельно. Проверка маршрутов:

```bash
curl -fsS https://muravyinaya-ferma.online/health
curl -fsS https://api.muravyinaya-ferma.online/health
curl -fsS https://seif.176-108-251-123.sslip.io/health
systemctl --user is-active seif-cloudflare-tunnel.service seif-cloudru-tunnel.service
```

`setup-kind-node.sh` настраивает NVIDIA Runtime/CDI только внутри узла kind и
локальный TCP-прокси для NodePort. Docker daemon хоста он не перезапускает.
При пересоздании узла повторите подготовку и загрузку образов. PVC и данные
внутри старого kind-узла при его удалении не сохраняются.

Это одноузловой стенд. Два GPU pod разделяют одну видеокарту без аппаратной
изоляции видеопамяти; Kubernetes здесь не учитывает `nvidia.com/gpu` при
планировании. Реплики Redis/Sentinel переживают отказ отдельного pod, но не
потерю узла. HPA требует Metrics Server. Прежние результаты на 2000 RPS
нельзя приписывать этому новому сетевому пути без отдельного замера.

# Локальный Kubernetes (kind)

Этот overlay запускает API из коммита `fcf76c5`, две CPU NER-реплики и три Redis/Sentinel-пода на одном kind-узле. Он нужен для проверки Kubernetes-развёртывания и отказа отдельного пода. GPU RuBERT и измеренные 2000 RPS относятся к отдельному профилю `deploy/rubert` и этим overlay не подтверждаются.

Создайте кластер с именем `seif` и соберите образы из коммита `fcf76c5` (теги фиксируют проверенную версию кода):

```bash
kind create cluster --name seif --wait 5m
docker build -t seif-api:local-fcf76c5 .
docker build -t seif-ner:local-fcf76c5 -f Dockerfile.ner .
kind load docker-image seif-api:local-fcf76c5 seif-ner:local-fcf76c5 --name seif
```

Локальные PV используют каталоги внутри kind-узла. Создайте их с правами UID Redis:

```bash
docker exec seif-control-plane sh -c 'mkdir -p /var/local/seif/redis-0 /var/local/seif/redis-1 /var/local/seif/redis-2 && chown -R 999:999 /var/local/seif'
kubectl apply -f deploy/k8s/base/namespace.yaml
python3 deploy/k8s/create-secrets.py
kubectl apply -k deploy/k8s/overlays/local
kubectl -n seif patch configmap redis-bootstrap --type merge -p '{"data":{"allow":"true"}}'
kubectl -n seif rollout status statefulset/redis --timeout=10m
kubectl -n seif patch configmap redis-bootstrap --type merge -p '{"data":{"allow":"false"}}'
kubectl -n seif rollout status deployment/seif-ner --timeout=10m
kubectl -n seif rollout status deployment/seif-api --timeout=10m
```

Получите API-ключ из Secret локально и направьте запрос через `kubectl -n seif port-forward --address 127.0.0.1 svc/seif-api 8877:80`. Адрес проверки: `http://127.0.0.1:8877/health`. Для `/process` используйте заголовки `X-System-ID: bank-assistant` и `X-API-Key`. Не публикуйте значения Secret в логах или репозитории.

Это одноузловая среда: PV переживают перезапуск пода, но удаляются вместе с kind-узлом; отказоустойчивость к потере узла здесь не проверяется. kind CNI не обеспечивает проверку NetworkPolicy, а HPA без Metrics Server не получает CPU-метрики. Production-манифесты и процедура многозонного запуска описаны в `docs/kubernetes.md`.

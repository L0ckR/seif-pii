#!/usr/bin/env bash
# Prepare an existing WSL2 kind node to run SEIF's local RuBERT profile.
set -euo pipefail

model_dir=${1:?Pass the RuBERT checkpoint directory as the first argument}
node=${2:-seif-control-plane}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT

test -d "$model_dir"
docker exec "$node" test -c /dev/dxg
nvidia-ctk cdi generate --output "$scratch/nvidia.yaml"
python3 - "$scratch/nvidia.yaml" "$scratch/transfer.list" <<'PY'
import sys
from pathlib import Path

import yaml

spec = yaml.safe_load(Path(sys.argv[1]).read_text())
paths = [mount["hostPath"] for mount in spec["containerEdits"]["mounts"]]
paths.extend(("/usr/bin/nvidia-container-runtime", "/usr/bin/nvidia-cdi-hook"))
Path(sys.argv[2]).write_text("\n".join(path.lstrip("/") for path in paths) + "\n")
PY

docker exec -u 0 "$node" mkdir -p /etc/cdi /etc/nvidia-container-runtime
tar -C / -cf - -T "$scratch/transfer.list" | docker exec -i -u 0 "$node" tar -C / -xf -
docker cp "$scratch/nvidia.yaml" "$node:/etc/cdi/nvidia.yaml"
sed 's/mode = "auto"/mode = "cdi"/' /etc/nvidia-container-runtime/config.toml > "$scratch/runtime.toml"
docker cp "$scratch/runtime.toml" "$node:/etc/nvidia-container-runtime/config.toml"

if ! docker exec "$node" grep -q 'runtimes.nvidia]' /etc/containerd/config.toml; then
  docker exec -u 0 "$node" cp /etc/containerd/config.toml /etc/containerd/config.toml.pre-seif-gpu
  docker exec -i -u 0 "$node" sh -c 'cat >> /etc/containerd/config.toml' <<'EOF'
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"
  base_runtime_spec = "/etc/containerd/cri-base.json"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
    SystemdCgroup = true
EOF
  docker exec -u 0 "$node" systemctl restart containerd
fi

docker exec -u 0 "$node" mkdir -p /var/local/seif/model /var/local/seif/capture
tar -C "$model_dir" -cf - . | docker exec -i -u 0 "$node" tar -C /var/local/seif/model -xf -
docker exec -u 0 "$node" chown -R 10001:10001 /var/local/seif/model /var/local/seif/capture
docker exec -u 0 "$node" chmod 700 /var/local/seif/capture

docker exec -u 0 "$node" sh -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends haproxy && rm -rf /var/lib/apt/lists/*'
docker cp "$script_dir/node-gateway.cfg" "$node:/etc/haproxy/seif-node-gateway.cfg"
docker cp "$script_dir/node-gateway.service" "$node:/etc/systemd/system/seif-node-gateway.service"
docker exec -u 0 "$node" systemctl daemon-reload
docker exec -u 0 "$node" systemctl enable --now seif-node-gateway.service
docker update --restart unless-stopped "$node" >/dev/null
echo "kind node prepared for SEIF GPU pods"

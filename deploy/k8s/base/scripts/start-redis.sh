#!/bin/sh
set -eu
. /config/common.sh

case "${MIN_REPLICAS_TO_WRITE:-1}" in 0|1|2) ;; *) exit 64 ;; esac
case "${MIN_REPLICAS_MAX_LAG:-5}" in ''|*[!0-9]*) exit 64 ;; esac
case "${REDIS_MAXMEMORY:-1gb}" in ''|*[!0-9mMgGbBkK]*) exit 64 ;; esac
mkdir -p /data/redis

echo 'Redis is waiting for agreement from two Sentinel endpoints.'
until master=$(discover_master 2); do sleep 3; done

cat > /data/redis/redis.conf <<EOF
bind 0.0.0.0
protected-mode yes
port 6379
daemonize no
logfile ""
dir /data/redis
requirepass "$REDIS_PASSWORD"
masterauth "$REDIS_PASSWORD"
replica-announce-ip "$SELF"
replica-announce-port 6379
replica-read-only yes
replica-serve-stale-data no
appendonly yes
appendfsync everysec
aof-use-rdb-preamble yes
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
save ""
maxmemory ${REDIS_MAXMEMORY:-1gb}
maxmemory-policy noeviction
min-replicas-to-write ${MIN_REPLICAS_TO_WRITE:-1}
min-replicas-max-lag ${MIN_REPLICAS_MAX_LAG:-5}
repl-backlog-size 16mb
repl-diskless-sync yes
repl-diskless-sync-delay 1
EOF

if [ "$master" != "$SELF" ]; then
  printf 'replicaof %s 6379\n' "$master" >> /data/redis/redis.conf
  echo "Starting Redis replica; current primary is $master."
else
  echo 'Starting Redis at the primary address agreed by Sentinel.'
fi
exec redis-server /data/redis/redis.conf

#!/bin/sh
set -eu
export REDISCLI_AUTH="${REDIS_PASSWORD:?}"
MASTER_NAME=${MASTER_NAME:-seif-master}

case "${1:-}" in
  redis-live)
    [ "$(redis-cli --raw -h 127.0.0.1 -p 6379 PING 2>/dev/null)" = PONG ]
    ;;
  sentinel-live)
    [ "$(redis-cli --raw -h 127.0.0.1 -p 26379 PING 2>/dev/null)" = PONG ]
    ;;
  sentinel-ready)
    reply=$(redis-cli --raw -h 127.0.0.1 -p 26379 SENTINEL CKQUORUM "$MASTER_NAME" 2>/dev/null)
    case "$reply" in OK*) exit 0 ;; *) exit 1 ;; esac
    ;;
  redis-ready)
    replication=$(redis-cli --raw -h 127.0.0.1 -p 6379 INFO replication 2>/dev/null | tr -d '\r')
    role=$(printf '%s\n' "$replication" | sed -n 's/^role://p')
    case "$role" in
      master)
        replicas=$(printf '%s\n' "$replication" | sed -n 's/^connected_slaves://p')
        [ "${replicas:-0}" -ge "${MIN_REPLICAS_TO_WRITE:-1}" ]
        ;;
      slave)
        link=$(printf '%s\n' "$replication" | sed -n 's/^master_link_status://p')
        [ "$link" = up ]
        ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 64 ;;
esac

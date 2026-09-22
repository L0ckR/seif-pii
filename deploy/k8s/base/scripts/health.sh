#!/bin/sh
set -eu
export REDISCLI_AUTH="${REDIS_PASSWORD:?}"
MASTER_NAME=${MASTER_NAME:-seif-master}

case "${1:-}" in
  redis-live)
    # PING is rejected by a stale replica when replica-serve-stale-data=no.
    # INFO is explicitly permitted during STALE/LOADING in Redis 7.4.
    server=$(redis-cli --raw -h 127.0.0.1 -p 6379 INFO server 2>/dev/null)
    case "$server" in *redis_version:*) exit 0 ;; *) exit 1 ;; esac
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
        syncing=$(printf '%s\n' "$replication" | sed -n 's/^master_sync_in_progress://p')
        [ "$link" = up ] && [ "$syncing" = 0 ]
        ;;
      *) exit 1 ;;
    esac
    ;;
  *) exit 64 ;;
esac

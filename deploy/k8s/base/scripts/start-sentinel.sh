#!/bin/sh
set -eu
. /config/common.sh

mkdir -p /data/sentinel
configuration=/data/sentinel/sentinel.conf
marker=/data/sentinel/member-initialized

if [ -s "$configuration" ]; then
  # Keep Sentinel's identity, epochs, elected primary and discovered peers.
  # Password refresh reads the environment; no credential enters argv or logs.
  awk 'BEGIN { pw = ENVIRON["REDIS_PASSWORD"] }
    /^requirepass / { print "requirepass \"" pw "\""; next }
    /^sentinel auth-pass / { print "sentinel auth-pass " $3 " \"" pw "\""; next }
    { print }' "$configuration" > "$configuration.next"
  mv "$configuration.next" "$configuration"
  exec redis-server "$configuration" --sentinel
fi

echo 'Sentinel is waiting for an existing cluster or explicit initial bootstrap.'
while :; do
  # A replacement member needs two existing Sentinels. A single peer is enough
  # only while the operator explicitly authorizes first cluster formation.
  required=2
  if bootstrap_allowed && [ ! -e "$marker" ]; then required=1; fi
  if master=$(discover_master "$required"); then break; fi
  if [ "$POD_NAME" = redis-0 ] && [ ! -e "$marker" ] && bootstrap_allowed; then
    master=$(peer_name 0)
    break
  fi
  sleep 3
done

# A headless Service publishes unready addresses; wait for its DNS record.
until nslookup "$master" >/dev/null 2>&1; do sleep 2; done
cat > "$configuration.next" <<EOF
bind 0.0.0.0
protected-mode yes
port 26379
daemonize no
logfile ""
dir /data/sentinel
requirepass "$REDIS_PASSWORD"
sentinel resolve-hostnames yes
sentinel announce-hostnames yes
sentinel announce-ip "$SELF"
sentinel announce-port 26379
sentinel monitor $MASTER_NAME $master 6379 2
sentinel auth-pass $MASTER_NAME "$REDIS_PASSWORD"
sentinel down-after-milliseconds $MASTER_NAME 10000
sentinel failover-timeout $MASTER_NAME 60000
sentinel parallel-syncs $MASTER_NAME 1
sentinel deny-scripts-reconfig yes
EOF
mv "$configuration.next" "$configuration"
touch "$marker"
exec redis-server "$configuration" --sentinel

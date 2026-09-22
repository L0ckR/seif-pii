#!/usr/bin/env python3
"""Save authorized live-vault originals locally without modifying the serving app.

Only run when retention is permitted. Files contain plaintext personal data and
belong under ignored local-data/. No inputs, keys or exception details are logged.
Read a healthy Redis replica; never issue writes, MONITOR or configuration changes.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from redis import Redis  # noqa: E402
from redis.sentinel import Sentinel  # noqa: E402

from scripts.benchmark import SYNTHETIC_PAYLOADS  # noqa: E402
from seif.config import Settings  # noqa: E402
from seif.transform import restore_exact  # noqa: E402


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def restore_verified(cipher, hash_key, key, blob):
    """Authenticate ciphertext and prove reconstruction, including unmask-disabled records."""
    record = json.loads(cipher.decrypt(blob[:12], blob[12:], key.encode()))
    original = restore_exact(record)
    digest = hmac.new(hash_key, original.encode('utf-8'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, record['original_hash']):
        raise ValueError('Original is not reconstructible')
    return original, record, digest


def check_replica(client):
    state = client.info('replication')
    if (state.get('role') != 'slave' or state.get('master_link_status') != 'up'
            or state.get('master_sync_in_progress') != 0):
        raise RuntimeError('Capture requires a healthy replica')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60, help='Bounded observation in seconds, at most 900')
    parser.add_argument('--max-bytes', type=int, default=256 * 1024 * 1024)
    parser.add_argument('--max-records', type=int, default=100000)
    args = parser.parse_args()
    if not 0 < args.duration <= 900 or args.max_bytes < 1024 or not 0 < args.max_records <= 1000000:
        parser.error('Invalid capture bounds')
    os.umask(0o077)
    load_dotenv(ROOT / '.env', override=False)
    settings = Settings.from_env()
    if not settings.sentinels:
        parser.error('This capture tool requires configured Sentinel replicas')
    sentinel = Sentinel(settings.sentinels, sentinel_kwargs={
        'password': settings.sentinel_password or None, 'socket_timeout': 2, 'socket_connect_timeout': 2,
    }, password=settings.redis_password or None, socket_timeout=2, socket_connect_timeout=2)
    replicas = sentinel.discover_slaves(settings.sentinel_master)
    if not replicas:
        raise RuntimeError('No replica available')
    host, port = replicas[0]
    client = Redis(host=host, port=port, password=settings.redis_password or None,
                   socket_timeout=2, socket_connect_timeout=2, decode_responses=False)
    check_replica(client)
    base = ROOT / 'local-data' / 'hackathon-requests'
    if (ROOT / 'local-data').is_symlink() or base.is_symlink():
        raise RuntimeError('Capture directory must not be a symlink')
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory.mkdir(mode=0o700)
    capture = directory / 'requests.jsonl'
    cipher = AESGCM(settings.encryption_key)
    hash_key = hmac.digest(settings.encryption_key, b'SEIF:hmac:v1', 'sha256')
    seen_keys, sources, frequencies = set(), {}, Counter()
    stats, started = Counter(), time.monotonic()
    last_role_check = started
    stop_reason, finished = 'duration', False
    manifest = {
        'schema_version': 1, 'started_at_utc': utcnow(), 'source': 'live_shared_vault_replica',
        'source_attribution': 'unverified: shared vault also contains local tests and browser demos',
        'user_confirmed_retention_permission': True, 'plaintext': True,
        'original_payload_ids_available': False,
        'annotations': 'Observed results/types are model predictions, NOT gold labels',
        'timing': 'observed_at is capture time; original arrival times, HTTP statuses and retries are unavailable',
        'retention': 'Local manual retention; vault TTL unchanged; delete this directory when no longer needed',
    }

    def save_manifest():
        data = {**manifest, 'updated_at_utc': utcnow(), 'complete': finished,
                'stop_reason': stop_reason, 'counts': dict(stats), 'source_records_per_case': dict(frequencies)}
        temp = directory / 'manifest.next.json'
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temp.replace(directory / 'manifest.json')

    print(json.dumps({'capture_directory': str(directory), 'status': 'capturing'}), flush=True)
    try:
        with capture.open('x', encoding='utf-8') as stream:
            while time.monotonic() - started < args.duration:
                check_replica(client)
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor=cursor, match='seif:v1:*', count=100)
                    for offset in range(0, len(keys), 8):
                        if time.monotonic() - last_role_check >= 2:
                            check_replica(client)
                            last_role_check = time.monotonic()
                        batch = [key for key in keys[offset:offset + 8] if key not in seen_keys]
                        if not batch:
                            continue
                        if len(seen_keys) + len(batch) > args.max_records:
                            stop_reason = 'record_limit'
                            break
                        blobs = client.mget(batch)
                        for key, blob in zip(batch, blobs, strict=True):
                            seen_keys.add(key)
                            stats['source_records_seen'] += 1
                            if blob is None:
                                stats['expired_before_read'] += 1
                                continue
                            try:
                                original, record, digest = restore_verified(cipher, hash_key, key.decode(), blob)
                            except Exception:
                                stats['unrecoverable_records'] += 1
                                continue
                            if digest in sources:
                                frequencies[sources[digest]] += 1
                                stats['duplicate_originals'] += 1
                                continue
                            case_id = f'capture_{len(sources) + 1:06d}'
                            row = {'case_id': case_id, 'payload': original,
                                   'observed_result': record['masked'],
                                   'observed_types': sorted({s['type'] for s in record['entities']}),
                                   'detector_profile': record.get('detector_profile', 'rules'),
                                   'known_local_benchmark_example': original in SYNTHETIC_PAYLOADS,
                                   'observed_at': utcnow()}
                            line = json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                            size = len(line.encode('utf-8'))
                            if stats['jsonl_bytes'] + size > args.max_bytes:
                                stop_reason = 'byte_limit'
                                break
                            stream.write(line)
                            sources[digest] = case_id
                            frequencies[case_id] = 1
                            stats['unique_originals'] += 1
                            stats['jsonl_bytes'] += size
                            stats['known_local_benchmark_examples'] += int(row['known_local_benchmark_example'])
                        stream.flush()
                        if stop_reason != 'duration':
                            break
                        time.sleep(.003)
                    if stop_reason != 'duration' or cursor == 0 or time.monotonic() - started >= args.duration:
                        break
                    if stats['source_records_seen'] % 1000 < 100:
                        save_manifest()
                stats['scan_passes'] += 1
                save_manifest()
                if stop_reason != 'duration':
                    break
                time.sleep(min(2, max(0, args.duration - (time.monotonic() - started))))
            os.fsync(stream.fileno())
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
    except Exception:
        stop_reason = 'capture_read_error'
        stats['read_failures'] += 1
    finally:
        client.close()
        finished = True
        save_manifest()
    print(json.dumps({'capture_directory': str(directory), 'stop_reason': stop_reason,
                      'counts': dict(stats)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Capture did not start; check local configuration and replica availability.', file=sys.stderr)
        raise SystemExit(1) from None

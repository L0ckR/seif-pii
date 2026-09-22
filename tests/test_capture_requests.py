"""Capture validation uses only synthetic records and never connects to Redis."""
import hmac
from types import SimpleNamespace

import pytest
from cryptography.exceptions import InvalidTag

from scripts.capture_requests import check_replica, restore_verified
from seif.config import Settings
from seif.transform import mask
from seif.vault import Vault


def sealed_record(mode="mask", allow_unmask=True):
    vault = Vault(Settings(encryption_key=bytes(range(32))))
    original = "🧪 ФИО: Тестов Макет Макетович; почта: demo@example.invalid."
    person = "Тестов Макет Макетович"
    email = "demo@example.invalid"
    spans = [
        SimpleNamespace(type=kind, start=original.index(value), end=original.index(value) + len(value))
        for kind, value in (("PERSON", person), ("EMAIL", email))
    ]
    masked, replacements = mask(original, spans, mode)
    record = {
        "masked": masked,
        "replacements": replacements if allow_unmask else [],
        "original_hash": vault.digest(original),
    }
    key = vault.key("synthetic-capture-test", "synthetic-id")
    return vault, original, key, vault._seal(key, record)


@pytest.mark.parametrize("mode", ["mask", "token", "synthetic"])
def test_capture_restores_exact_unicode_across_transform_modes(mode):
    vault, original, key, blob = sealed_record(mode)

    restored, record, digest = restore_verified(vault.cipher, vault.hash_key, key, blob)

    assert restored == original
    assert record["masked"] != original
    assert hmac.compare_digest(digest, vault.digest(original))


def test_capture_rejects_ciphertext_tampering():
    vault, _, key, blob = sealed_record()
    corrupted = blob[:-1] + bytes([blob[-1] ^ 1])

    with pytest.raises(InvalidTag):
        restore_verified(vault.cipher, vault.hash_key, key, corrupted)


def test_capture_binds_ciphertext_to_its_redis_key():
    vault, _, key, blob = sealed_record()

    with pytest.raises(InvalidTag):
        restore_verified(vault.cipher, vault.hash_key, key + "different", blob)


def test_capture_rejects_unmask_disabled_record_instead_of_saving_mask_as_original():
    vault, _, key, blob = sealed_record(allow_unmask=False)

    with pytest.raises(ValueError, match="Original is not reconstructible"):
        restore_verified(vault.cipher, vault.hash_key, key, blob)


def test_capture_verifies_hash_even_for_authenticated_record():
    vault, _, key, blob = sealed_record()
    record = vault._open(key, blob)
    record["original_hash"] = "0" * 64
    authenticated_but_inconsistent = vault._seal(key, record)

    with pytest.raises(ValueError, match="Original is not reconstructible"):
        restore_verified(vault.cipher, vault.hash_key, key, authenticated_but_inconsistent)


class ReplicaStub:
    def __init__(self, state):
        self.state = state
        self.commands = []

    def info(self, section):
        self.commands.append(("INFO", section))
        return self.state


def test_replica_guard_accepts_only_healthy_replica_with_read_only_inspection():
    client = ReplicaStub({"role": "slave", "master_link_status": "up", "master_sync_in_progress": 0})

    assert check_replica(client) is None
    assert client.commands == [("INFO", "replication")]


@pytest.mark.parametrize("state", [
    {"role": "master", "master_link_status": "up", "master_sync_in_progress": 0},
    {"role": "slave", "master_link_status": "down", "master_sync_in_progress": 0},
    {"role": "slave", "master_link_status": "up", "master_sync_in_progress": 1},
    {"role": "slave", "master_link_status": "up"},
    {},
])
def test_replica_guard_rejects_primary_stale_syncing_or_unknown_state(state):
    client = ReplicaStub(state)

    with pytest.raises(RuntimeError, match="healthy replica"):
        check_replica(client)
    assert client.commands == [("INFO", "replication")]

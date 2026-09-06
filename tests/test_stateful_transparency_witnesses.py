from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tiny_tensor_compiler import (
    NativeBundleTransparencyError,
    NativeBundleTransparencyRollbackError,
    TransparencyStateStore,
    TransparencyWitnessPolicy,
    create_release_checkpoint,
    create_stateful_transparency_witness_quorum,
    create_transparency_checkpoint,
    publisher_public_key_from_private_key,
    verify_transparency_checkpoint,
    verify_transparency_witness_quorum,
)


def _key(seed: int) -> bytes:
    return bytes([seed]) * 32


def _public(seed: int) -> bytes:
    return publisher_public_key_from_private_key(_key(seed))


def _release(sequence: int) -> bytes:
    return create_release_checkpoint(
        _key(7),
        "stable",
        sequence,
        f"sha256:{sequence:064x}",
    )


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(size: int) -> int:
    return 1 << ((size - 1).bit_length() - 1)


def _root(leaves: list[bytes]) -> bytes:
    if len(leaves) == 1:
        return _leaf_hash(leaves[0])
    split = _split(len(leaves))
    return _node_hash(_root(leaves[:split]), _root(leaves[split:]))


def _subproof(old_size: int, leaves: list[bytes], complete: bool) -> list[bytes]:
    if old_size == len(leaves):
        return [] if complete else [_root(leaves)]
    split = _split(len(leaves))
    if old_size <= split:
        return _subproof(old_size, leaves[:split], complete) + [_root(leaves[split:])]
    return _subproof(old_size - split, leaves[split:], False) + [_root(leaves[:split])]


def _consistency_proof(old_size: int, leaves: list[bytes]) -> tuple[bytes, ...]:
    return tuple(_subproof(old_size, leaves, True))


def _checkpoint(log_private: bytes, leaves: list[bytes]) -> bytes:
    return create_transparency_checkpoint(log_private, len(leaves), _root(leaves))


def _stores(tmp_path: Path, log_public: bytes) -> tuple[TransparencyStateStore, ...]:
    return (
        TransparencyStateStore(tmp_path / "witness-1.json", log_public),
        TransparencyStateStore(tmp_path / "witness-2.json", log_public),
    )


def test_stateful_quorum_records_each_witness_before_returning_signature(
    tmp_path: Path,
) -> None:
    log_private = _key(90)
    log_public = _public(90)
    leaves = [_release(1)]
    encoded_checkpoint = _checkpoint(log_private, leaves)
    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public)
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    stores = _stores(tmp_path, log_public)

    encoded_quorum = create_stateful_transparency_witness_quorum(
        (_key(1), _key(2)),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_stores=stores,
        consistency_proofs=((), ()),
    )

    verified = verify_transparency_witness_quorum(encoded_quorum, encoded_checkpoint, policy)
    assert len(verified.witness_ids) == 2
    assert all(store.current() == checkpoint for store in stores)


def test_stateful_quorum_advances_only_after_valid_consistency_for_every_signer(
    tmp_path: Path,
) -> None:
    log_private = _key(91)
    log_public = _public(91)
    releases = [_release(sequence) for sequence in (1, 2, 3, 4)]
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    stores = _stores(tmp_path, log_public)
    head1_encoded = _checkpoint(log_private, releases[:1])
    head1 = verify_transparency_checkpoint(head1_encoded, log_public)
    for store in stores:
        store.record(head1)

    head4_encoded = _checkpoint(log_private, releases)
    head4 = verify_transparency_checkpoint(head4_encoded, log_public)
    valid = _consistency_proof(1, releases)
    damaged = (*valid[:-1], b"\xff" * 32)

    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            head4_encoded,
            log_public_key=log_public,
            state_stores=stores,
            consistency_proofs=(valid, damaged),
        )
    assert all(store.current() == head1 for store in stores)

    create_stateful_transparency_witness_quorum(
        (_key(1), _key(2)),
        policy,
        head4_encoded,
        log_public_key=log_public,
        state_stores=stores,
        consistency_proofs=(valid, valid),
    )
    assert all(store.current() == head4 for store in stores)


def test_stateful_quorum_rejects_rollback_and_same_size_fork_without_advancing(
    tmp_path: Path,
) -> None:
    log_private = _key(92)
    log_public = _public(92)
    releases = [_release(sequence) for sequence in (1, 2, 3)]
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    stores = _stores(tmp_path, log_public)
    head3_encoded = _checkpoint(log_private, releases)
    head3 = verify_transparency_checkpoint(head3_encoded, log_public)
    for store in stores:
        store.record(head3)

    head2_encoded = _checkpoint(log_private, releases[:2])
    with pytest.raises(NativeBundleTransparencyRollbackError, match="rollback"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            head2_encoded,
            log_public_key=log_public,
            state_stores=stores,
            consistency_proofs=((), ()),
        )
    assert all(store.current() == head3 for store in stores)

    fork = [*releases[:2], create_release_checkpoint(_key(8), "stable", 99, f"sha256:{99:064x}")]
    fork_encoded = _checkpoint(log_private, fork)
    with pytest.raises(NativeBundleTransparencyError, match="same-size transparency fork"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            fork_encoded,
            log_public_key=log_public,
            state_stores=stores,
            consistency_proofs=((), ()),
        )
    assert all(store.current() == head3 for store in stores)


def test_stateful_quorum_rejects_shared_state_and_wrong_log_before_advancing(
    tmp_path: Path,
) -> None:
    log_private = _key(93)
    log_public = _public(93)
    encoded_checkpoint = _checkpoint(log_private, [_release(1)])
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    shared = TransparencyStateStore(tmp_path / "shared.json", log_public)

    with pytest.raises(ValueError, match="distinct state"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            encoded_checkpoint,
            log_public_key=log_public,
            state_stores=(shared, shared),
            consistency_proofs=((), ()),
        )
    assert shared.current() is None

    wrong_log_public = _public(94)
    wrong_store = TransparencyStateStore(tmp_path / "wrong-log.json", wrong_log_public)
    right_store = TransparencyStateStore(tmp_path / "right-log.json", log_public)
    with pytest.raises(NativeBundleTransparencyError, match="pinned log operator"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            encoded_checkpoint,
            log_public_key=log_public,
            state_stores=(right_store, wrong_store),
            consistency_proofs=((), ()),
        )
    assert right_store.current() is None
    assert wrong_store.current() is None


def test_stateful_quorum_rejects_invalid_checkpoint_signature_before_state_change(
    tmp_path: Path,
) -> None:
    log_private = _key(95)
    log_public = _public(95)
    encoded_checkpoint = bytearray(_checkpoint(log_private, [_release(1)]))
    signature_marker = b'"signature":"'
    start = encoded_checkpoint.index(signature_marker) + len(signature_marker)
    encoded_checkpoint[start] = ord("0") if encoded_checkpoint[start] != ord("0") else ord("1")
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    stores = _stores(tmp_path, log_public)

    with pytest.raises(NativeBundleTransparencyError, match="signature verification failed"):
        create_stateful_transparency_witness_quorum(
            (_key(1), _key(2)),
            policy,
            bytes(encoded_checkpoint),
            log_public_key=log_public,
            state_stores=stores,
            consistency_proofs=((), ()),
        )
    assert all(store.current() is None for store in stores)

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tiny_tensor_compiler import (
    NativeBundleTransparencyError,
    NativeBundleTransparencyRollbackError,
    PublisherTrustPolicy,
    TransparencyStateStore,
    accept_release_transparency,
    create_release_checkpoint,
    create_transparency_checkpoint,
    log_id_from_public_key,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
    verify_release_checkpoint,
    verify_transparency_checkpoint,
    verify_transparency_consistency,
    verify_transparency_inclusion,
)


def _key(seed: int) -> bytes:
    return bytes([seed]) * 32


def _release(seed: int, sequence: int) -> bytes:
    digest = f"sha256:{sequence:064x}"
    return create_release_checkpoint(_key(seed), "stable", sequence, digest)


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(size: int) -> int:
    return 1 << ((size - 1).bit_length() - 1)


def _root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return _leaf_hash(leaves[0])
    split = _split(len(leaves))
    return _node_hash(_root(leaves[:split]), _root(leaves[split:]))


def _inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    if len(leaves) == 1:
        return []
    split = _split(len(leaves))
    if index < split:
        return _inclusion_proof(leaves[:split], index) + [_root(leaves[split:])]
    return _inclusion_proof(leaves[split:], index - split) + [_root(leaves[:split])]


def _subproof(old_size: int, leaves: list[bytes], complete: bool) -> list[bytes]:
    if old_size == len(leaves):
        return [] if complete else [_root(leaves)]
    split = _split(len(leaves))
    if old_size <= split:
        return _subproof(old_size, leaves[:split], complete) + [_root(leaves[split:])]
    return _subproof(old_size - split, leaves[split:], False) + [_root(leaves[:split])]


def _consistency_proof(old_size: int, leaves: list[bytes]) -> list[bytes]:
    return _subproof(old_size, leaves, True)


def _log_keys(seed: int = 91) -> tuple[bytes, bytes]:
    private = _key(seed)
    return private, publisher_public_key_from_private_key(private)


def _checkpoint(private: bytes, leaves: list[bytes]) -> bytes:
    return create_transparency_checkpoint(private, len(leaves), _root(leaves))


def test_accepts_authenticated_release_inclusion_and_append_only_extension(tmp_path: Path) -> None:
    publisher_private = _key(7)
    publisher_public = publisher_public_key_from_private_key(publisher_private)
    publisher_id = publisher_id_from_public_key(publisher_public)
    policy = PublisherTrustPolicy((publisher_public,))
    releases = [_release(7, sequence) for sequence in (1, 2, 3)]

    # Transparency composes with, but never replaces, the existing publisher verifier.
    verified = verify_release_checkpoint(
        releases[0],
        policy,
        expected_publisher=publisher_id,
        expected_channel="stable",
    )
    assert verified.sequence == 1

    log_private, log_public = _log_keys()
    store = TransparencyStateStore(tmp_path / "transparency.json", log_public)

    first_encoded = _checkpoint(log_private, releases[:1])
    first = accept_release_transparency(
        releases[0],
        leaf_index=0,
        encoded_checkpoint=first_encoded,
        log_public_key=log_public,
        inclusion_proof=_inclusion_proof(releases[:1], 0),
        consistency_proof=(),
        state_store=store,
    )
    assert first.tree_size == 1
    assert store.current() == first

    third_encoded = _checkpoint(log_private, releases)
    third = accept_release_transparency(
        releases[2],
        leaf_index=2,
        encoded_checkpoint=third_encoded,
        log_public_key=log_public,
        inclusion_proof=_inclusion_proof(releases, 2),
        consistency_proof=_consistency_proof(1, releases),
        state_store=store,
    )
    assert third.tree_size == 3
    assert store.current() == third

    reopened = TransparencyStateStore(store.path, log_public)
    assert reopened.current() == third


def test_rfc6962_inclusion_verifier_rejects_wrong_leaf_index_and_proof() -> None:
    releases = [_release(3, sequence) for sequence in range(1, 6)]
    log_private, log_public = _log_keys(92)
    checkpoint = verify_transparency_checkpoint(_checkpoint(log_private, releases), log_public)

    verify_transparency_inclusion(
        releases[3],
        leaf_index=3,
        checkpoint=checkpoint,
        proof=_inclusion_proof(releases, 3),
    )

    with pytest.raises(NativeBundleTransparencyError, match="inclusion proof"):
        verify_transparency_inclusion(
            releases[3],
            leaf_index=2,
            checkpoint=checkpoint,
            proof=_inclusion_proof(releases, 3),
        )

    damaged = list(_inclusion_proof(releases, 3))
    damaged[0] = bytes([damaged[0][0] ^ 1]) + damaged[0][1:]
    with pytest.raises(NativeBundleTransparencyError, match="inclusion proof"):
        verify_transparency_inclusion(
            releases[3],
            leaf_index=3,
            checkpoint=checkpoint,
            proof=damaged,
        )


def test_rfc6962_consistency_verifier_accepts_non_power_of_two_growth() -> None:
    releases = [_release(4, sequence) for sequence in range(1, 8)]
    log_private, log_public = _log_keys(93)
    old = verify_transparency_checkpoint(_checkpoint(log_private, releases[:3]), log_public)
    new = verify_transparency_checkpoint(_checkpoint(log_private, releases), log_public)

    verify_transparency_consistency(old, new, _consistency_proof(3, releases))

    damaged = list(_consistency_proof(3, releases))
    damaged[-1] = b"\xff" * 32
    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        verify_transparency_consistency(old, new, damaged)


def test_state_rejects_rollback_and_same_size_fork_without_overwriting_floor(tmp_path: Path) -> None:
    releases = [_release(5, sequence) for sequence in (1, 2, 3)]
    log_private, log_public = _log_keys(94)
    store = TransparencyStateStore(tmp_path / "transparency.json", log_public)

    head3 = verify_transparency_checkpoint(_checkpoint(log_private, releases), log_public)
    store.record(head3)

    head2 = verify_transparency_checkpoint(_checkpoint(log_private, releases[:2]), log_public)
    with pytest.raises(NativeBundleTransparencyRollbackError, match="rollback"):
        store.record(head2, consistency_proof=_consistency_proof(2, releases))
    assert store.current() == head3

    fork_leaves = [*releases[:2], _release(6, 99)]
    fork = verify_transparency_checkpoint(_checkpoint(log_private, fork_leaves), log_public)
    with pytest.raises(NativeBundleTransparencyError, match="same-size transparency fork"):
        store.record(fork)
    assert store.current() == head3

    # Re-accepting the exact same signed head is idempotent and needs no proof.
    assert store.record(head3) == head3


def test_state_requires_valid_consistency_before_advancing(tmp_path: Path) -> None:
    releases = [_release(8, sequence) for sequence in (1, 2, 3, 4)]
    log_private, log_public = _log_keys(95)
    store = TransparencyStateStore(tmp_path / "transparency.json", log_public)
    head1 = verify_transparency_checkpoint(_checkpoint(log_private, releases[:1]), log_public)
    head4 = verify_transparency_checkpoint(_checkpoint(log_private, releases), log_public)
    store.record(head1)

    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        store.record(head4, consistency_proof=())
    assert store.current() == head1

    assert store.record(head4, consistency_proof=_consistency_proof(1, releases)) == head4


def test_checkpoint_signature_canonicality_and_operator_binding(tmp_path: Path) -> None:
    release = _release(9, 1)
    log_private, log_public = _log_keys(96)
    encoded = _checkpoint(log_private, [release])
    checkpoint = verify_transparency_checkpoint(encoded, log_public)

    assert checkpoint.log_id == log_id_from_public_key(log_public)
    assert checkpoint.tree_size == 1

    _, wrong_public = _log_keys(97)
    with pytest.raises(NativeBundleTransparencyError, match="log operator"):
        verify_transparency_checkpoint(encoded, wrong_public)

    noncanonical = encoded.replace(b'"schema"', b' "schema"', 1)
    with pytest.raises(NativeBundleTransparencyError, match="canonical"):
        verify_transparency_checkpoint(noncanonical, log_public)

    store = TransparencyStateStore(tmp_path / "state.json", log_public)
    store.record(checkpoint)
    wrong_store = TransparencyStateStore(store.path, wrong_public)
    with pytest.raises(NativeBundleTransparencyError, match="pinned log operator"):
        wrong_store.current()


def test_proof_and_argument_bounds_fail_closed(tmp_path: Path) -> None:
    release = _release(10, 1)
    log_private, log_public = _log_keys(98)
    checkpoint = verify_transparency_checkpoint(_checkpoint(log_private, [release]), log_public)

    with pytest.raises(ValueError, match="leaf index"):
        verify_transparency_inclusion(release, leaf_index=1, checkpoint=checkpoint, proof=())
    with pytest.raises(NativeBundleTransparencyError, match="proof node"):
        verify_transparency_inclusion(
            release,
            leaf_index=0,
            checkpoint=checkpoint,
            proof=(b"short",),
        )
    with pytest.raises(ValueError, match="tree size"):
        create_transparency_checkpoint(log_private, 0, _root([]))
    with pytest.raises(ValueError, match="exactly 32"):
        TransparencyStateStore(tmp_path / "state.json", b"short")

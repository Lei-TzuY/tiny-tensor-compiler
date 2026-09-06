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
    create_transparency_checkpoint,
    publisher_public_key_from_private_key,
    verify_transparency_checkpoint,
)
from tiny_tensor_compiler.native_bundle_attestation import publisher_id_from_public_key
from tiny_tensor_compiler.native_bundle_transparency_witness import (
    NativeBundleTransparencyWitnessError,
)
from tiny_tensor_compiler.native_bundle_transparency_witness_evidence import (
    TransparencyWitnessEvidenceStore,
)
from tiny_tensor_compiler.native_bundle_transparency_witness_observation import (
    create_transparency_witness_observation,
    verify_transparency_witness_observation,
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


def _observation(
    tmp_path: Path,
    *,
    state_name: str,
    witness_seed: int,
    log_private: bytes,
    policy: TransparencyWitnessPolicy,
    leaves: list[bytes],
) -> bytes:
    log_public = publisher_public_key_from_private_key(log_private)
    encoded_checkpoint = _checkpoint(log_private, leaves)
    witness_state = TransparencyStateStore(tmp_path / state_name, log_public)
    witness_state.record(verify_transparency_checkpoint(encoded_checkpoint, log_public))
    return create_transparency_witness_observation(
        _key(witness_seed),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_store=witness_state,
    )


def _digest(
    observation: bytes,
    *,
    log_public: bytes,
    policy: TransparencyWitnessPolicy,
) -> str:
    return verify_transparency_witness_observation(
        observation,
        log_public_key=log_public,
        policy=policy,
    ).checkpoint_digest


def test_evidence_store_persists_verified_latest_observations_and_reopens(tmp_path: Path) -> None:
    log_private = _key(110)
    log_public = _public(110)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    observation = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    path = tmp_path / "evidence.json"
    store = TransparencyWitnessEvidenceStore(path, log_public, policy)

    snapshot = store.record(observation)
    reopened = TransparencyWitnessEvidenceStore(path, log_public, policy).current()

    assert snapshot.status == "healthy"
    assert snapshot.fork_evidence is None
    assert len(snapshot.observations) == 1
    assert reopened == snapshot
    assert snapshot.observations[0].encoded_observation == observation


def test_evidence_store_requires_pairwise_growth_proofs_and_deduplicates_checkpoint_digest(
    tmp_path: Path,
) -> None:
    log_private = _key(111)
    log_public = _public(111)
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    leaves = [_release(index) for index in (1, 2, 3, 4)]
    obs1 = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    obs2 = _observation(
        tmp_path,
        state_name="w2.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    obs3 = _observation(
        tmp_path,
        state_name="w3.json",
        witness_seed=3,
        log_private=log_private,
        policy=policy,
        leaves=leaves,
    )
    old_digest = _digest(obs1, log_public=log_public, policy=policy)
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(obs1)
    store.record(obs2)

    snapshot = store.record(
        obs3,
        consistency_proofs={old_digest: _consistency_proof(1, leaves)},
    )

    assert snapshot.status == "healthy"
    assert tuple(item.witness_id for item in snapshot.observations) == tuple(
        sorted((publisher_id_from_public_key(_public(1)), publisher_id_from_public_key(_public(2)), publisher_id_from_public_key(_public(3))))
    )
    assert [item.checkpoint.tree_size for item in snapshot.observations].count(1) == 2
    assert [item.checkpoint.tree_size for item in snapshot.observations].count(4) == 1


def test_evidence_store_rejects_missing_damaged_and_extra_growth_proofs_without_mutation(
    tmp_path: Path,
) -> None:
    log_private = _key(112)
    log_public = _public(112)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3)]
    old = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    new = _observation(
        tmp_path,
        state_name="w2.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=leaves,
    )
    old_digest = _digest(old, log_public=log_public, policy=policy)
    path = tmp_path / "evidence.json"
    store = TransparencyWitnessEvidenceStore(path, log_public, policy)
    baseline = store.record(old)

    with pytest.raises(NativeBundleTransparencyWitnessError, match="consistency proof set"):
        store.record(new)
    assert store.current() == baseline

    proof = _consistency_proof(1, leaves)
    damaged = (*proof[:-1], b"\xff" * 32)
    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        store.record(new, consistency_proofs={old_digest: damaged})
    assert store.current() == baseline

    with pytest.raises(NativeBundleTransparencyWitnessError, match="consistency proof set"):
        store.record(
            new,
            consistency_proofs={
                old_digest: proof,
                "sha256:" + "00" * 32: proof,
            },
        )
    assert store.current() == baseline


def test_evidence_store_rejects_per_witness_rollback(tmp_path: Path) -> None:
    log_private = _key(113)
    log_public = _public(113)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3)]
    old = _observation(
        tmp_path,
        state_name="w1-old.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    new = _observation(
        tmp_path,
        state_name="w1-new.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves,
    )
    old_digest = _digest(old, log_public=log_public, policy=policy)
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(old)
    advanced = store.record(
        new,
        consistency_proofs={old_digest: _consistency_proof(1, leaves)},
    )

    with pytest.raises(NativeBundleTransparencyRollbackError, match="rollback"):
        store.record(old)
    assert store.current() == advanced


def test_evidence_store_persists_terminal_cross_witness_fork_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    log_private = _key(114)
    log_public = _public(114)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    first = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    conflicting = _observation(
        tmp_path,
        state_name="w2.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=[_release(99)],
    )
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(first)

    forked = store.record(conflicting)
    reopened = TransparencyWitnessEvidenceStore(
        tmp_path / "evidence.json", log_public, policy
    ).current()

    assert forked.status == "forked"
    assert forked.observations == ()
    assert forked.fork_evidence is not None
    assert reopened == forked
    left, right = forked.fork_evidence
    assert left.checkpoint.tree_size == right.checkpoint.tree_size == 1
    assert left.checkpoint.root_hash != right.checkpoint.root_hash

    with pytest.raises(NativeBundleTransparencyWitnessError, match="terminal fork"):
        store.record(first)
    assert store.current() == forked


def test_evidence_store_also_persists_same_witness_equivocation(tmp_path: Path) -> None:
    log_private = _key(115)
    log_public = _public(115)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    first = _observation(
        tmp_path,
        state_name="w1-a.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    conflicting = _observation(
        tmp_path,
        state_name="w1-b.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(2)],
    )
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(first)
    forked = store.record(conflicting)

    assert forked.status == "forked"
    assert forked.fork_evidence is not None
    assert forked.fork_evidence[0].witness_id == forked.fork_evidence[1].witness_id
    assert forked.fork_evidence[0].checkpoint.root_hash != forked.fork_evidence[1].checkpoint.root_hash


def test_evidence_store_two_instances_merge_under_shared_state_lock(tmp_path: Path) -> None:
    log_private = _key(116)
    log_public = _public(116)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    checkpoint_leaves = [_release(1), _release(2)]
    obs1 = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=checkpoint_leaves,
    )
    obs2 = _observation(
        tmp_path,
        state_name="w2.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=checkpoint_leaves,
    )
    path = tmp_path / "evidence.json"
    first_store = TransparencyWitnessEvidenceStore(path, log_public, policy)
    second_store = TransparencyWitnessEvidenceStore(path, log_public, policy)

    first_store.record(obs1)
    second_store.record(obs2)
    snapshot = first_store.current()

    assert snapshot is not None
    assert snapshot.status == "healthy"
    assert len(snapshot.observations) == 2


def test_evidence_store_rejects_tampered_state_and_wrong_pinned_policy(tmp_path: Path) -> None:
    log_private = _key(117)
    log_public = _public(117)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    observation = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    path = tmp_path / "evidence.json"
    store = TransparencyWitnessEvidenceStore(path, log_public, policy)
    store.record(observation)

    wrong_policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    with pytest.raises(NativeBundleTransparencyWitnessError, match="policy identity"):
        TransparencyWitnessEvidenceStore(path, log_public, wrong_policy).current()

    path.write_bytes(path.read_bytes().replace(b'"healthy"', b'"broken"', 1))
    with pytest.raises(NativeBundleTransparencyWitnessError, match="state"):
        store.current()

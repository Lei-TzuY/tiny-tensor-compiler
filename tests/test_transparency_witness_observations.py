from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tiny_tensor_compiler import (
    NativeBundleTransparencyError,
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
from tiny_tensor_compiler.native_bundle_transparency_witness_observation import (
    compare_transparency_witness_observations,
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


def _record(
    tmp_path: Path,
    name: str,
    log_public: bytes,
    encoded_checkpoint: bytes,
) -> TransparencyStateStore:
    store = TransparencyStateStore(tmp_path / name, log_public)
    store.record(verify_transparency_checkpoint(encoded_checkpoint, log_public))
    return store


def test_observation_requires_current_persisted_state_and_round_trips(tmp_path: Path) -> None:
    log_private = _key(90)
    log_public = _public(90)
    encoded_checkpoint = _checkpoint(log_private, [_release(1)])
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    store = TransparencyStateStore(tmp_path / "w1.json", log_public)

    with pytest.raises(NativeBundleTransparencyWitnessError, match="accepted checkpoint"):
        create_transparency_witness_observation(
            _key(1),
            policy,
            encoded_checkpoint,
            log_public_key=log_public,
            state_store=store,
        )

    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public)
    store.record(checkpoint)
    encoded = create_transparency_witness_observation(
        _key(1),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_store=store,
    )
    verified = verify_transparency_witness_observation(
        encoded,
        log_public_key=log_public,
        policy=policy,
    )

    witness1_id = publisher_id_from_public_key(_public(1))
    assert verified.policy_id == policy.policy_id
    assert verified.witness_id == witness1_id
    assert verified.checkpoint == checkpoint
    assert verified.encoded_checkpoint == encoded_checkpoint
    assert verified.checkpoint_digest == f"sha256:{hashlib.sha256(encoded_checkpoint).hexdigest()}"

    newer = _checkpoint(log_private, [_release(1), _release(2)])
    with pytest.raises(NativeBundleTransparencyWitnessError, match="current checkpoint"):
        create_transparency_witness_observation(
            _key(1),
            policy,
            newer,
            log_public_key=log_public,
            state_store=store,
        )


def test_cross_witness_same_checkpoint_agrees_deterministically(tmp_path: Path) -> None:
    log_private = _key(91)
    log_public = _public(91)
    encoded_checkpoint = _checkpoint(log_private, [_release(1), _release(2)])
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    store1 = _record(tmp_path, "w1.json", log_public, encoded_checkpoint)
    store2 = _record(tmp_path, "w2.json", log_public, encoded_checkpoint)
    obs1 = create_transparency_witness_observation(
        _key(1), policy, encoded_checkpoint, log_public_key=log_public, state_store=store1
    )
    obs2 = create_transparency_witness_observation(
        _key(2), policy, encoded_checkpoint, log_public_key=log_public, state_store=store2
    )

    comparison = compare_transparency_witness_observations(
        obs2,
        obs1,
        log_public_key=log_public,
        policy=policy,
    )

    assert comparison.relation == "same_checkpoint"
    assert comparison.first.witness_id < comparison.second.witness_id
    assert comparison.first.checkpoint == comparison.second.checkpoint
    assert comparison.policy_id == policy.policy_id


def test_cross_witness_same_size_fork_returns_signed_equivocation_evidence(
    tmp_path: Path,
) -> None:
    log_private = _key(92)
    log_public = _public(92)
    checkpoint_a = _checkpoint(log_private, [_release(1)])
    checkpoint_b = _checkpoint(log_private, [_release(99)])
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    store1 = _record(tmp_path, "w1.json", log_public, checkpoint_a)
    store2 = _record(tmp_path, "w2.json", log_public, checkpoint_b)
    obs1 = create_transparency_witness_observation(
        _key(1), policy, checkpoint_a, log_public_key=log_public, state_store=store1
    )
    obs2 = create_transparency_witness_observation(
        _key(2), policy, checkpoint_b, log_public_key=log_public, state_store=store2
    )

    comparison = compare_transparency_witness_observations(
        obs1,
        obs2,
        log_public_key=log_public,
        policy=policy,
    )

    assert comparison.relation == "same_size_fork"
    assert comparison.first.checkpoint.tree_size == comparison.second.checkpoint.tree_size == 1
    assert comparison.first.checkpoint.root_hash != comparison.second.checkpoint.root_hash
    assert comparison.first.encoded_observation in {obs1, obs2}
    assert comparison.second.encoded_observation in {obs1, obs2}


def test_cross_witness_growth_requires_exact_consistency_evidence(tmp_path: Path) -> None:
    log_private = _key(93)
    log_public = _public(93)
    leaves = [_release(sequence) for sequence in (1, 2, 3, 4)]
    head1 = _checkpoint(log_private, leaves[:1])
    head4 = _checkpoint(log_private, leaves)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    store1 = _record(tmp_path, "w1.json", log_public, head1)
    store2 = _record(tmp_path, "w2.json", log_public, head4)
    obs1 = create_transparency_witness_observation(
        _key(1), policy, head1, log_public_key=log_public, state_store=store1
    )
    obs2 = create_transparency_witness_observation(
        _key(2), policy, head4, log_public_key=log_public, state_store=store2
    )
    proof = _consistency_proof(1, leaves)

    comparison = compare_transparency_witness_observations(
        obs2,
        obs1,
        log_public_key=log_public,
        policy=policy,
        consistency_proof=proof,
    )
    assert comparison.relation == "consistent_growth"
    assert comparison.first.checkpoint.tree_size == 1
    assert comparison.second.checkpoint.tree_size == 4

    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        compare_transparency_witness_observations(
            obs1,
            obs2,
            log_public_key=log_public,
            policy=policy,
        )

    damaged = (*proof[:-1], b"\xff" * 32)
    with pytest.raises(NativeBundleTransparencyError, match="consistency proof"):
        compare_transparency_witness_observations(
            obs1,
            obs2,
            log_public_key=log_public,
            policy=policy,
            consistency_proof=damaged,
        )


def test_observation_rejects_tamper_wrong_policy_and_revoked_witness(tmp_path: Path) -> None:
    log_private = _key(94)
    log_public = _public(94)
    encoded_checkpoint = _checkpoint(log_private, [_release(1)])
    witness1_id = publisher_id_from_public_key(_public(1))
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    store = _record(tmp_path, "w1.json", log_public, encoded_checkpoint)
    encoded = bytearray(
        create_transparency_witness_observation(
            _key(1),
            policy,
            encoded_checkpoint,
            log_public_key=log_public,
            state_store=store,
        )
    )
    marker = b'"signature":"'
    start = encoded.index(marker) + len(marker)
    encoded[start] = ord("0") if encoded[start] != ord("0") else ord("1")
    with pytest.raises(NativeBundleTransparencyWitnessError, match="signature verification failed"):
        verify_transparency_witness_observation(
            bytes(encoded),
            log_public_key=log_public,
            policy=policy,
        )

    wrong_policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=2)
    original = create_transparency_witness_observation(
        _key(1),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_store=store,
    )
    with pytest.raises(NativeBundleTransparencyWitnessError, match="policy identity"):
        verify_transparency_witness_observation(
            original,
            log_public_key=log_public,
            policy=wrong_policy,
        )

    revoked = TransparencyWitnessPolicy(
        (_public(1), _public(2)),
        threshold=1,
        revoked_witnesses=frozenset({witness1_id}),
    )
    with pytest.raises(NativeBundleTransparencyWitnessError, match="revoked"):
        create_transparency_witness_observation(
            _key(1),
            revoked,
            encoded_checkpoint,
            log_public_key=log_public,
            state_store=store,
        )


def test_cross_witness_comparison_rejects_self_comparison_and_irrelevant_proof(
    tmp_path: Path,
) -> None:
    log_private = _key(95)
    log_public = _public(95)
    encoded_checkpoint = _checkpoint(log_private, [_release(1)])
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    store = _record(tmp_path, "w1.json", log_public, encoded_checkpoint)
    observation = create_transparency_witness_observation(
        _key(1),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_store=store,
    )

    with pytest.raises(NativeBundleTransparencyWitnessError, match="distinct witnesses"):
        compare_transparency_witness_observations(
            observation,
            observation,
            log_public_key=log_public,
            policy=policy,
        )

    store2 = _record(tmp_path, "w2.json", log_public, encoded_checkpoint)
    observation2 = create_transparency_witness_observation(
        _key(2),
        policy,
        encoded_checkpoint,
        log_public_key=log_public,
        state_store=store2,
    )
    with pytest.raises(NativeBundleTransparencyWitnessError, match="same-size"):
        compare_transparency_witness_observations(
            observation,
            observation2,
            log_public_key=log_public,
            policy=policy,
            consistency_proof=(b"\x00" * 32,),
        )

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiny_tensor_compiler import (
    NativeBundleTransparencyWitnessError,
    TransparencyStateStore,
    TransparencyWitnessPolicy,
    accept_witnessed_release_transparency,
    create_release_checkpoint,
    create_transparency_checkpoint,
    create_transparency_witness_quorum,
    publisher_public_key_from_private_key,
    verify_transparency_witness_quorum,
)


def _key(seed: int) -> bytes:
    return bytes([seed]) * 32


def _public(seed: int) -> bytes:
    return publisher_public_key_from_private_key(_key(seed))


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _release(sequence: int) -> bytes:
    return create_release_checkpoint(
        _key(7),
        "stable",
        sequence,
        f"sha256:{sequence:064x}",
    )


def _checkpoint(log_private: bytes, release: bytes) -> bytes:
    return create_transparency_checkpoint(log_private, 1, _leaf_hash(release))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def test_witness_policy_and_quorum_are_deterministic() -> None:
    policy = TransparencyWitnessPolicy((_public(3), _public(1), _public(2)), threshold=2)
    reordered = TransparencyWitnessPolicy((_public(2), _public(3), _public(1)), threshold=2)
    checkpoint = _checkpoint(_key(90), _release(1))

    encoded = create_transparency_witness_quorum((_key(2), _key(1)), policy, checkpoint)
    reversed_encoded = create_transparency_witness_quorum((_key(1), _key(2)), policy, checkpoint)
    verified = verify_transparency_witness_quorum(encoded, checkpoint, policy)

    assert policy.policy_id == reordered.policy_id
    assert encoded == reversed_encoded
    assert verified.policy_id == policy.policy_id
    assert verified.checkpoint_digest == f"sha256:{hashlib.sha256(checkpoint).hexdigest()}"
    assert verified.witness_ids == tuple(sorted(verified.witness_ids))
    assert len(verified.witness_ids) == 2


def test_quorum_creation_rejects_insufficient_duplicate_and_unknown_witnesses() -> None:
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    checkpoint = _checkpoint(_key(91), _release(1))

    with pytest.raises(NativeBundleTransparencyWitnessError, match="at least 2"):
        create_transparency_witness_quorum((_key(1),), policy, checkpoint)
    with pytest.raises(NativeBundleTransparencyWitnessError, match="duplicate witness"):
        create_transparency_witness_quorum((_key(1), _key(1)), policy, checkpoint)
    with pytest.raises(NativeBundleTransparencyWitnessError, match="not in the witness policy"):
        create_transparency_witness_quorum((_key(1), _key(9)), policy, checkpoint)


def test_quorum_is_bound_to_exact_checkpoint_bytes() -> None:
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    checkpoint = _checkpoint(_key(92), _release(1))
    other_checkpoint = _checkpoint(_key(92), _release(2))
    encoded = create_transparency_witness_quorum((_key(1), _key(2)), policy, checkpoint)

    with pytest.raises(NativeBundleTransparencyWitnessError, match="checkpoint digest"):
        verify_transparency_witness_quorum(encoded, other_checkpoint, policy)


def test_quorum_rejects_tampered_signature() -> None:
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    checkpoint = _checkpoint(_key(93), _release(1))
    encoded = create_transparency_witness_quorum((_key(1), _key(2)), policy, checkpoint)
    envelope = json.loads(encoded.decode("ascii"))
    signature = envelope["signatures"][0]["signature"]
    envelope["signatures"][0]["signature"] = ("0" if signature[0] != "0" else "1") + signature[1:]

    with pytest.raises(NativeBundleTransparencyWitnessError, match="signature verification failed"):
        verify_transparency_witness_quorum(_canonical(envelope), checkpoint, policy)


def test_revoked_witness_does_not_count_toward_threshold() -> None:
    base_policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    checkpoint = _checkpoint(_key(94), _release(1))
    encoded = create_transparency_witness_quorum((_key(1), _key(2)), base_policy, checkpoint)
    revoked_policy = TransparencyWitnessPolicy(
        (_public(1), _public(2), _public(3)),
        threshold=2,
        revoked_witnesses=frozenset((base_policy.witness_ids[0],)),
    )

    with pytest.raises(NativeBundleTransparencyWitnessError, match="valid non-revoked"):
        verify_transparency_witness_quorum(encoded, checkpoint, revoked_policy)


def test_witness_policy_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="threshold"):
        TransparencyWitnessPolicy((_public(1), _public(2)), threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        TransparencyWitnessPolicy((_public(1), _public(2)), threshold=3)
    with pytest.raises(ValueError, match="duplicate"):
        TransparencyWitnessPolicy((_public(1), _public(1)), threshold=1)


def test_witness_gate_fails_before_transparency_state_advances(tmp_path: Path) -> None:
    release = _release(1)
    log_private = _key(95)
    log_public = _public(95)
    checkpoint = _checkpoint(log_private, release)
    state = TransparencyStateStore(tmp_path / "transparency.json", log_public)
    policy = TransparencyWitnessPolicy((_public(1), _public(2), _public(3)), threshold=2)
    wrong_policy = TransparencyWitnessPolicy((_public(4), _public(5), _public(6)), threshold=2)
    quorum = create_transparency_witness_quorum((_key(1), _key(2)), policy, checkpoint)

    with pytest.raises(NativeBundleTransparencyWitnessError, match="policy identity"):
        accept_witnessed_release_transparency(
            release,
            leaf_index=0,
            encoded_checkpoint=checkpoint,
            log_public_key=log_public,
            inclusion_proof=(),
            consistency_proof=(),
            state_store=state,
            encoded_witness_quorum=quorum,
            witness_policy=wrong_policy,
        )
    assert state.current() is None

    accepted = accept_witnessed_release_transparency(
        release,
        leaf_index=0,
        encoded_checkpoint=checkpoint,
        log_public_key=log_public,
        inclusion_proof=(),
        consistency_proof=(),
        state_store=state,
        encoded_witness_quorum=quorum,
        witness_policy=policy,
    )
    assert accepted.tree_size == 1
    assert state.current() == accepted

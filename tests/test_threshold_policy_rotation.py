from __future__ import annotations

import json
from pathlib import Path

import pytest


def _key(seed: int) -> bytes:
    return bytes((seed + index) % 256 for index in range(32))


def _policy(seeds: tuple[int, ...], threshold: int = 2, revoked=frozenset()):
    from tiny_tensor_compiler.native_bundle_attestation import (
        publisher_public_key_from_private_key,
    )
    from tiny_tensor_compiler.native_bundle_threshold import ThresholdReleasePolicy

    public_keys = tuple(publisher_public_key_from_private_key(_key(seed)) for seed in seeds)
    return ThresholdReleasePolicy(public_keys, threshold, revoked)


def test_transition_is_canonical_order_independent_and_binds_pinned_next_policy() -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        create_threshold_policy_transition,
        verify_threshold_policy_transition,
    )

    current = _policy((1, 11, 21))
    next_policy = _policy((31, 41, 51))
    encoded = create_threshold_policy_transition(
        (_key(11), _key(1)),
        current,
        next_policy,
        1,
    )
    transition = verify_threshold_policy_transition(
        encoded,
        current,
        next_policy,
        expected_epoch=1,
    )
    assert transition.from_policy_id == current.policy_id
    assert transition.to_policy_id == next_policy.policy_id
    assert transition.epoch == 1
    decoded = json.loads(encoded)
    assert [item["signer_id"] for item in decoded["signatures"]] == sorted(
        item["signer_id"] for item in decoded["signatures"]
    )
    assert encoded.endswith(b"\n")

    different_next = _policy((61, 71, 81))
    with pytest.raises(Exception, match="pinned next policy"):
        verify_threshold_policy_transition(encoded, current, different_next)


def test_transition_requires_current_threshold_members_and_distinct_signatures() -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        NativeBundlePolicyRotationError,
        create_threshold_policy_transition,
    )

    current = _policy((1, 11, 21))
    next_policy = _policy((31, 41, 51))
    with pytest.raises(NativeBundlePolicyRotationError, match="at least 2"):
        create_threshold_policy_transition((_key(1),), current, next_policy, 1)
    with pytest.raises(NativeBundlePolicyRotationError, match="duplicate signer"):
        create_threshold_policy_transition((_key(1), _key(1)), current, next_policy, 1)
    with pytest.raises(Exception, match="not in the threshold policy"):
        create_threshold_policy_transition((_key(1), _key(99)), current, next_policy, 1)


def test_transition_honors_current_policy_local_revocation() -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        NativeBundlePolicyRotationError,
        create_threshold_policy_transition,
        verify_threshold_policy_transition,
    )

    base = _policy((1, 11, 21))
    revoked_id = base.signer_ids[0]
    current = _policy((1, 11, 21), revoked=frozenset({revoked_id}))
    next_policy = _policy((31, 41, 51))

    key_by_id = {}
    from tiny_tensor_compiler.native_bundle_attestation import (
        publisher_id_from_public_key,
        publisher_public_key_from_private_key,
    )

    for seed in (1, 11, 21):
        private = _key(seed)
        public = publisher_public_key_from_private_key(private)
        key_by_id[publisher_id_from_public_key(public)] = private
    eligible = [signer for signer in current.signer_ids if signer != revoked_id]
    encoded = create_threshold_policy_transition(
        tuple(key_by_id[signer] for signer in eligible),
        current,
        next_policy,
        1,
    )
    assert verify_threshold_policy_transition(encoded, current, next_policy).epoch == 1

    with pytest.raises(NativeBundlePolicyRotationError, match="revoked"):
        create_threshold_policy_transition(
            (key_by_id[revoked_id], key_by_id[eligible[0]]),
            current,
            next_policy,
            1,
        )


def test_rotation_state_advances_forward_and_new_policy_authorizes_release(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        ThresholdPolicyRotationStateStore,
        create_threshold_policy_transition,
    )
    from tiny_tensor_compiler.native_bundle_threshold import (
        create_threshold_release_checkpoint,
        verify_threshold_release_checkpoint,
    )

    bootstrap = _policy((1, 11, 21))
    next_policy = _policy((31, 41, 51))
    store = ThresholdPolicyRotationStateStore(tmp_path / "rotation.json", bootstrap)
    assert store.current() == (0, bootstrap.policy_id)

    encoded = create_threshold_policy_transition(
        (_key(1), _key(11)),
        bootstrap,
        next_policy,
        1,
    )
    accepted = store.accept_transition(encoded, bootstrap, next_policy)
    assert accepted.epoch == 1
    assert store.current() == (1, next_policy.policy_id)

    checkpoint = create_threshold_release_checkpoint(
        (_key(31), _key(41)),
        next_policy,
        "stable",
        9,
        "sha256:" + "ab" * 32,
    )
    assert verify_threshold_release_checkpoint(checkpoint, next_policy).sequence == 9

    reopened = ThresholdPolicyRotationStateStore(tmp_path / "rotation.json", bootstrap)
    assert reopened.current() == (1, next_policy.policy_id)


def test_rotation_state_rejects_replay_epoch_skip_and_fork(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        NativeBundlePolicyRotationError,
        ThresholdPolicyRotationStateStore,
        create_threshold_policy_transition,
    )

    bootstrap = _policy((1, 11, 21))
    first = _policy((31, 41, 51))
    fork = _policy((61, 71, 81))
    second = _policy((91, 101, 111))
    store = ThresholdPolicyRotationStateStore(tmp_path / "rotation.json", bootstrap)

    to_first = create_threshold_policy_transition((_key(1), _key(11)), bootstrap, first, 1)
    to_fork = create_threshold_policy_transition((_key(1), _key(11)), bootstrap, fork, 1)
    store.accept_transition(to_first, bootstrap, first)

    with pytest.raises(NativeBundlePolicyRotationError, match="current threshold policy"):
        store.accept_transition(to_first, bootstrap, first)
    with pytest.raises(NativeBundlePolicyRotationError, match="current threshold policy"):
        store.accept_transition(to_fork, bootstrap, fork)

    skipped = create_threshold_policy_transition((_key(31), _key(41)), first, second, 3)
    with pytest.raises(NativeBundlePolicyRotationError, match="epoch must be exactly 2"):
        store.accept_transition(skipped, first, second)
    assert store.current() == (1, first.policy_id)


def test_transition_tampering_and_wrong_predecessor_fail_closed() -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        NativeBundlePolicyRotationError,
        create_threshold_policy_transition,
        verify_threshold_policy_transition,
    )

    bootstrap = _policy((1, 11, 21))
    next_policy = _policy((31, 41, 51))
    wrong_current = _policy((61, 71, 81))
    encoded = create_threshold_policy_transition(
        (_key(1), _key(11)),
        bootstrap,
        next_policy,
        1,
    )
    tampered = encoded.replace(b'"epoch":1', b'"epoch":2')
    with pytest.raises(NativeBundlePolicyRotationError, match="signature verification"):
        verify_threshold_policy_transition(tampered, bootstrap, next_policy)
    with pytest.raises(NativeBundlePolicyRotationError, match="predecessor"):
        verify_threshold_policy_transition(encoded, wrong_current, next_policy)


def test_rotation_state_rejects_wrong_bootstrap_and_corrupt_state(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        NativeBundlePolicyRotationError,
        ThresholdPolicyRotationStateStore,
        create_threshold_policy_transition,
    )

    bootstrap = _policy((1, 11, 21))
    next_policy = _policy((31, 41, 51))
    path = tmp_path / "rotation.json"
    store = ThresholdPolicyRotationStateStore(path, bootstrap)
    encoded = create_threshold_policy_transition((_key(1), _key(11)), bootstrap, next_policy, 1)
    store.accept_transition(encoded, bootstrap, next_policy)

    with pytest.raises(NativeBundlePolicyRotationError, match="bootstrap"):
        ThresholdPolicyRotationStateStore(path, _policy((61, 71, 81)))

    path.write_text('{"schema":"broken"}\n', encoding="ascii")
    with pytest.raises(NativeBundlePolicyRotationError, match="fields"):
        ThresholdPolicyRotationStateStore(path, bootstrap)


def test_transition_validation_rejects_invalid_epoch_and_identity_noop() -> None:
    from tiny_tensor_compiler.native_bundle_policy_rotation import (
        ThresholdPolicyTransition,
        create_threshold_policy_transition,
    )

    policy = _policy((1, 11, 21))
    with pytest.raises(ValueError, match="at least 1"):
        create_threshold_policy_transition((_key(1), _key(11)), policy, _policy((31, 41, 51)), 0)
    with pytest.raises(ValueError, match="change policy identity"):
        ThresholdPolicyTransition(policy.policy_id, policy.policy_id, 1)

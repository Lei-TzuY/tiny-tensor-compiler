from __future__ import annotations

import json

import pytest


def _key(seed: int) -> bytes:
    return bytes((seed + index) % 256 for index in range(32))


def test_archive_attestation_is_deterministic_and_policy_verified() -> None:
    from tiny_tensor_compiler import (
        PublisherTrustPolicy,
        create_archive_attestation,
        publisher_id_from_public_key,
        publisher_public_key_from_private_key,
        verify_archive_attestation,
    )

    secret = _key(0)
    digest = "sha256:" + "1" * 64
    public = publisher_public_key_from_private_key(secret)
    publisher = publisher_id_from_public_key(public)
    policy = PublisherTrustPolicy((public,))
    attestation = create_archive_attestation(secret, digest)

    assert create_archive_attestation(secret, digest) == attestation
    assert verify_archive_attestation(attestation, digest, policy) == publisher
    assert policy.trusted_publishers == (publisher,)


def test_archive_attestation_rejects_modified_signature_and_digest() -> None:
    from tiny_tensor_compiler import (
        NativeBundleTrustError,
        PublisherTrustPolicy,
        create_archive_attestation,
        publisher_public_key_from_private_key,
        verify_archive_attestation,
    )

    secret = _key(0)
    digest = "sha256:" + "2" * 64
    policy = PublisherTrustPolicy((publisher_public_key_from_private_key(secret),))
    attestation = create_archive_attestation(secret, digest)
    decoded = json.loads(attestation)
    signature = decoded["signature"]
    decoded["signature"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    modified = (
        json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )

    with pytest.raises(NativeBundleTrustError, match="signature verification failed"):
        verify_archive_attestation(modified, digest, policy)
    with pytest.raises(NativeBundleTrustError, match="archive digest does not match"):
        verify_archive_attestation(attestation, "sha256:" + "3" * 64, policy)


def test_publisher_policy_rejects_unknown_and_revoked_identity() -> None:
    from tiny_tensor_compiler import (
        NativeBundleTrustError,
        PublisherTrustPolicy,
        publisher_id_from_public_key,
        publisher_public_key_from_private_key,
    )

    public = publisher_public_key_from_private_key(_key(0))
    other = publisher_public_key_from_private_key(_key(9))
    publisher = publisher_id_from_public_key(public)

    with pytest.raises(NativeBundleTrustError, match="not trusted"):
        PublisherTrustPolicy((other,)).public_key_for(publisher)
    with pytest.raises(NativeBundleTrustError, match="revoked"):
        PublisherTrustPolicy(
            (public,), revoked_publishers=frozenset({publisher})
        ).public_key_for(publisher)


def test_attestation_key_and_envelope_validation_fail_closed() -> None:
    from tiny_tensor_compiler import (
        NativeBundleTrustError,
        PublisherTrustPolicy,
        create_archive_attestation,
        publisher_public_key_from_private_key,
        verify_archive_attestation,
    )

    digest = "sha256:" + "4" * 64
    with pytest.raises(ValueError, match="exactly 32"):
        create_archive_attestation(b"short", digest)

    secret = _key(0)
    public = publisher_public_key_from_private_key(secret)
    policy = PublisherTrustPolicy((public,))
    decoded = json.loads(create_archive_attestation(secret, digest))
    decoded["unexpected"] = "field"
    malformed = (
        json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    with pytest.raises(NativeBundleTrustError, match="fields are invalid"):
        verify_archive_attestation(malformed, digest, policy)

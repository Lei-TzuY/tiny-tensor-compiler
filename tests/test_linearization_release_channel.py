from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_bundle_attested_registry import _key, _linearization_archive
from test_bundle_release_channel import _archive, _server

import tiny_tensor_compiler as ttc


def publisher():
    private = _key(43)
    public = ttc.publisher_public_key_from_private_key(private)
    return private, ttc.PublisherTrustPolicy((public,))


def test_retained_release_round_trip_is_compiler_free_and_reuses_frozen_state(
    tmp_path: Path, monkeypatch,
) -> None:
    archive = _linearization_archive(tmp_path)
    private, policy = publisher()
    store = ttc.ReleaseStateStore(tmp_path / "release-state.json")
    with _server("secret") as (registry, remote):
        checkpoint = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 7, token="secret", allow_insecure_http=True,
        )
        assert ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 7, token="secret", allow_insecure_http=True,
        ) == checkpoint
        destination = tmp_path / "accepted.ttcla"
        assert ttc.fetch_linearization_release_channel_archive(
            registry, checkpoint.publisher_id, "stable", destination, policy, store,
            token="secret", allow_insecure_http=True,
        ) == checkpoint
        assert destination.read_bytes() == archive.read_bytes()
        assert store.floor(checkpoint.publisher_id, "stable") == checkpoint

        def no_compiler(*_args, **_kwargs):
            raise AssertionError("compiler lookup during retained release loading")

        monkeypatch.setattr(ttc.native_bundle, "_compiler_command", no_compiler)
        executable = ttc.load_linearization_release_channel_registry(
            registry, checkpoint.publisher_id, "stable", policy, store,
            token="secret", allow_insecure_http=True,
        )
        download = executable._download_root
        try:
            assert executable.checkpoint == checkpoint
            assert (executable.sequence, executable.channel) == (7, "stable")
            assert executable.digest == checkpoint.archive_digest
            assert executable.publisher_id == checkpoint.publisher_id
            assert executable.symbolic_dims == ("B",)
            assert executable.available_bindings == ((("B", 2),), (("B", 5),))
            for batch in (2, 5):
                values = np.arange(batch * 3, dtype=np.float64).reshape(batch, 3) / 4 - 1.5
                frozen = values.copy()
                state = executable((values,))
                values[:] = 999
                np.testing.assert_array_equal(state.primal, np.array(np.sum(frozen**4)))
                for factor in (1.0, -0.5):
                    tangent = np.full_like(frozen, factor)
                    np.testing.assert_array_equal(
                        state.pushforward((tangent,)), np.array(np.sum(4 * frozen**3 * tangent))
                    )
                    np.testing.assert_array_equal(
                        state.pullback(np.array(factor)), 4 * frozen**3 * factor
                    )
            assert executable.loaded_bindings == executable.available_bindings
        finally:
            executable.close()
            executable.close()
        assert executable.closed and not download.exists()
        with pytest.raises(RuntimeError, match="closed"):
            executable.linearize((np.zeros((2, 3)),))
        assert all(auth == "Bearer secret" for _method, _path, auth in remote.requests)


def test_signed_retained_rollback_is_rejected_before_payload_requests(tmp_path: Path) -> None:
    archive = _linearization_archive(tmp_path)
    private, policy = publisher()
    state_path = tmp_path / "release-state.json"
    with _server() as (registry, remote):
        first = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 1, allow_insecure_http=True,
        )
        channel = next(path for path in remote.objects if path.startswith("/v1/channels/"))
        older_head = remote.objects[channel]
        newer = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 2, allow_insecure_http=True,
        )
        ttc.fetch_linearization_release_channel_archive(
            registry, first.publisher_id, "stable", tmp_path / "accepted.ttcla", policy,
            ttc.ReleaseStateStore(state_path), allow_insecure_http=True,
        )
        remote.channel_substitute = older_head
        before = len(remote.requests)
        rejected = tmp_path / "replayed.ttcla"
        with pytest.raises(ttc.NativeBundleRollbackError, match="rollback"):
            ttc.fetch_linearization_release_channel_archive(
                registry, first.publisher_id, "stable", rejected, policy,
                ttc.ReleaseStateStore(state_path), allow_insecure_http=True,
            )
        assert [path for _method, path, _auth in remote.requests[before:]] == [channel]
        assert not rejected.exists()
        assert ttc.ReleaseStateStore(state_path).floor(first.publisher_id, "stable") == newer


def test_retained_release_revocation_fails_before_network(tmp_path: Path) -> None:
    archive = _linearization_archive(tmp_path)
    private, policy = publisher()
    with _server() as (registry, remote):
        checkpoint = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 1, allow_insecure_http=True,
        )
        revoked = ttc.PublisherTrustPolicy(
            policy.public_keys, revoked_publishers=frozenset({checkpoint.publisher_id})
        )
        before = len(remote.requests)
        with pytest.raises(ttc.NativeBundleTrustError, match="revoked"):
            ttc.load_linearization_release_channel_registry(
                registry, checkpoint.publisher_id, "stable", revoked,
                ttc.ReleaseStateStore(tmp_path / "release-state.json"), allow_insecure_http=True,
            )
        assert len(remote.requests) == before


def test_wrong_payload_kind_does_not_advance_retained_release_floor(tmp_path: Path) -> None:
    archive = _archive(tmp_path, "ordinary", (2,))
    private, policy = publisher()
    store = ttc.ReleaseStateStore(tmp_path / "release-state.json")
    destination = tmp_path / "wrong-kind.ttcla"
    with _server() as (registry, _remote):
        checkpoint = ttc.publish_release_channel(
            archive, registry, private, "stable", 3, allow_insecure_http=True,
        )
        with pytest.raises(ttc.NativeBundleArchiveError, match="payload kind"):
            ttc.fetch_linearization_release_channel_archive(
                registry, checkpoint.publisher_id, "stable", destination, policy, store,
                allow_insecure_http=True,
            )
        assert store.floor(checkpoint.publisher_id, "stable") is None
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".wrong-kind.ttcla.release-*"))
        assert ttc.fetch_release_channel_archive(
            registry, checkpoint.publisher_id, "stable", tmp_path / "ordinary-downloaded.ttca", policy,
            store, allow_insecure_http=True,
        ) == checkpoint


def test_retained_bad_attestation_cannot_publish_or_advance_floor(tmp_path: Path) -> None:
    archive = _linearization_archive(tmp_path)
    private, policy = publisher()
    store = ttc.ReleaseStateStore(tmp_path / "release-state.json")
    destination = tmp_path / "bad-attestation.ttcla"
    with _server() as (registry, remote):
        checkpoint = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 9, allow_insecure_http=True,
        )
        path = next(path for path in remote.objects if path.startswith("/v1/attestations/"))
        remote.objects[path] = ttc.create_archive_attestation(_key(77), checkpoint.archive_digest)
        with pytest.raises(ttc.NativeBundleTrustError, match="identity mismatch"):
            ttc.fetch_linearization_release_channel_archive(
                registry, checkpoint.publisher_id, "stable", destination, policy, store,
                allow_insecure_http=True,
            )
        assert store.floor(checkpoint.publisher_id, "stable") is None
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".bad-attestation.ttcla.release-*"))


def test_retained_download_rechecks_concurrently_advanced_floor(
    tmp_path: Path, monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle_release

    archive = _linearization_archive(tmp_path)
    private, policy = publisher()
    store = ttc.ReleaseStateStore(tmp_path / "release-state.json")
    destination = tmp_path / "outdated.ttcla"
    with _server() as (registry, _remote):
        checkpoint = ttc.publish_linearization_release_channel(
            archive, registry, private, "stable", 10, allow_insecure_http=True,
        )
        newer = ttc.ReleaseCheckpoint(
            checkpoint.publisher_id, "stable", 11, checkpoint.archive_digest
        )
        fetch = native_bundle_release.fetch_attested_dynamic_linearization_bundle_set_archive

        def advance_after_fetch(*args, **kwargs):
            result = fetch(*args, **kwargs)
            ttc.ReleaseStateStore(store.path).record(newer)
            return result

        monkeypatch.setattr(
            native_bundle_release, "fetch_attested_dynamic_linearization_bundle_set_archive",
            advance_after_fetch,
        )
        with pytest.raises(ttc.NativeBundleRollbackError, match="rollback"):
            ttc.fetch_linearization_release_channel_archive(
                registry, checkpoint.publisher_id, "stable", destination, policy, store,
                allow_insecure_http=True,
            )
        assert store.floor(checkpoint.publisher_id, "stable") == newer
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".outdated.ttcla.release-*"))

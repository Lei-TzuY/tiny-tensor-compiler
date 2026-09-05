from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim


def _key(seed: int) -> bytes:
    return bytes((seed + index) % 256 for index in range(32))


class _State:
    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.channel_substitute: bytes | None = None


def _handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format, *_args) -> None:
            return

        def _authorized(self) -> bool:
            if state.token is None:
                return True
            return self.headers.get("Authorization") == f"Bearer {state.token}"

        def do_PUT(self) -> None:
            state.requests.append((self.command, self.path, self.headers.get("Authorization")))
            if not self._authorized():
                self.send_error(401)
                return
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0:
                self.send_error(411)
                return
            body = self.rfile.read(length)
            archive_prefix = "/v1/archives/sha256/"
            channel_prefix = "/v1/channels/threshold-ed25519/"
            immutable = self.path.startswith(archive_prefix)
            channel = self.path.startswith(channel_prefix)
            if immutable:
                expected = self.path.removeprefix(archive_prefix)
                if hashlib.sha256(body).hexdigest() != expected:
                    self.send_error(400)
                    return
            elif not channel:
                self.send_error(404)
                return

            current = state.objects.get(self.path)
            if immutable and current is not None:
                self.send_response(412)
                self.end_headers()
                return
            if channel:
                if self.headers.get("If-None-Match") == "*" and current is not None:
                    self.send_response(412)
                    self.end_headers()
                    return
                expected_etag = self.headers.get("If-Match")
                if expected_etag is not None and state.etags.get(self.path) != expected_etag:
                    self.send_response(412)
                    self.end_headers()
                    return
                if current is not None and expected_etag is None:
                    self.send_response(428)
                    self.end_headers()
                    return

            state.objects[self.path] = body
            etag = f'"{hashlib.sha256(body).hexdigest()}"'
            state.etags[self.path] = etag
            self.send_response(201 if current is None else 204)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            state.requests.append((self.command, self.path, self.headers.get("Authorization")))
            if not self._authorized():
                self.send_error(401)
                return
            if self.path not in state.objects:
                self.send_error(404)
                return
            body = state.objects[self.path]
            if state.channel_substitute is not None and self.path.startswith(
                "/v1/channels/threshold-ed25519/"
            ):
                body = state.channel_substitute
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", state.etags[self.path])
            self.end_headers()
            self.wfile.write(body)

    return Handler


@contextmanager
def _server(token: str | None = None):
    state = _State(token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _archive(tmp_path: Path, name: str, batches: tuple[int, ...]) -> Path:
    from tiny_tensor_compiler.native_bundle_archive import pack_dynamic_bundle_set_archive
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 3), dtype="float32")
    module = builder.finish((x.relu(), x + x))
    bundle = tmp_path / f"{name}.ttcset"
    archive = tmp_path / f"{name}.ttca"
    compile_dynamic_bundle_set(module, tuple({batch: value} for value in batches), bundle)
    pack_dynamic_bundle_set_archive(bundle, archive)
    return archive


def _policy(*, threshold: int = 2, revoked: frozenset[str] = frozenset()):
    from tiny_tensor_compiler.native_bundle_attestation import (
        publisher_public_key_from_private_key,
    )
    from tiny_tensor_compiler.native_bundle_threshold import ThresholdReleasePolicy

    public_keys = tuple(publisher_public_key_from_private_key(_key(seed)) for seed in (1, 11, 21))
    return ThresholdReleasePolicy(public_keys, threshold, revoked)


def test_threshold_policy_and_checkpoint_are_canonical_and_order_independent() -> None:
    from tiny_tensor_compiler.native_bundle_attestation import (
        publisher_public_key_from_private_key,
    )
    from tiny_tensor_compiler.native_bundle_threshold import (
        ThresholdReleasePolicy,
        create_threshold_release_checkpoint,
        verify_threshold_release_checkpoint,
    )

    keys = tuple(_key(seed) for seed in (1, 11, 21))
    public_keys = tuple(publisher_public_key_from_private_key(key) for key in keys)
    forward = ThresholdReleasePolicy(public_keys, 2)
    reverse = ThresholdReleasePolicy(tuple(reversed(public_keys)), 2)
    assert forward.policy_id == reverse.policy_id
    assert forward.signer_ids == reverse.signer_ids

    digest = "sha256:" + "ab" * 32
    encoded = create_threshold_release_checkpoint((keys[1], keys[0]), forward, "stable", 7, digest)
    checkpoint = verify_threshold_release_checkpoint(encoded, forward, expected_channel="stable")
    assert checkpoint.policy_id == forward.policy_id
    assert checkpoint.sequence == 7
    assert checkpoint.archive_digest == digest
    decoded = json.loads(encoded)
    assert [item["signer_id"] for item in decoded["signatures"]] == sorted(
        item["signer_id"] for item in decoded["signatures"]
    )
    assert encoded.endswith(b"\n")

    tampered = encoded.replace(b'"sequence":7', b'"sequence":6')
    with pytest.raises(Exception, match="signature"):
        verify_threshold_release_checkpoint(tampered, forward, expected_channel="stable")


def test_threshold_checkpoint_requires_distinct_policy_members_and_k_signatures() -> None:
    from tiny_tensor_compiler.native_bundle_threshold import (
        NativeBundleThresholdError,
        create_threshold_release_checkpoint,
    )

    policy = _policy()
    digest = "sha256:" + "cd" * 32
    with pytest.raises(NativeBundleThresholdError, match="at least 2"):
        create_threshold_release_checkpoint((_key(1),), policy, "stable", 1, digest)
    with pytest.raises(NativeBundleThresholdError, match="duplicate signer"):
        create_threshold_release_checkpoint((_key(1), _key(1)), policy, "stable", 1, digest)
    with pytest.raises(NativeBundleThresholdError, match="not in the threshold policy"):
        create_threshold_release_checkpoint((_key(1), _key(99)), policy, "stable", 1, digest)


def test_threshold_verification_applies_local_revocation_without_changing_policy_id() -> None:
    from tiny_tensor_compiler.native_bundle_attestation import publisher_id_from_public_key
    from tiny_tensor_compiler.native_bundle_threshold import (
        NativeBundleThresholdError,
        ThresholdReleasePolicy,
        create_threshold_release_checkpoint,
        verify_threshold_release_checkpoint,
    )

    base = _policy()
    signer_ids = base.signer_ids
    encoded_all = create_threshold_release_checkpoint(
        (_key(1), _key(11), _key(21)),
        base,
        "stable",
        4,
        "sha256:" + "ef" * 32,
    )
    revoked_one = ThresholdReleasePolicy(
        base.public_keys,
        2,
        frozenset({signer_ids[0]}),
    )
    assert revoked_one.policy_id == base.policy_id
    assert verify_threshold_release_checkpoint(encoded_all, revoked_one).sequence == 4

    key_by_id = {
        publisher_id_from_public_key(public): private
        for public, private in zip(
            (
                __import__(
                    "tiny_tensor_compiler.native_bundle_attestation",
                    fromlist=["publisher_public_key_from_private_key"],
                ).publisher_public_key_from_private_key(_key(seed))
                for seed in (1, 11, 21)
            ),
            (_key(1), _key(11), _key(21)),
            strict=True,
        )
    }
    revoked_id = signer_ids[0]
    other_id = next(signer for signer in signer_ids if signer != revoked_id)
    encoded_two = create_threshold_release_checkpoint(
        (key_by_id[revoked_id], key_by_id[other_id]),
        base,
        "stable",
        5,
        "sha256:" + "12" * 32,
    )
    with pytest.raises(NativeBundleThresholdError, match="found 1"):
        verify_threshold_release_checkpoint(encoded_two, revoked_one)


def test_threshold_state_rejects_rollback_and_same_sequence_equivocation(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_threshold import (
        NativeBundleThresholdError,
        NativeBundleThresholdRollbackError,
        ThresholdReleaseCheckpoint,
        ThresholdReleaseStateStore,
    )

    policy = _policy()
    store = ThresholdReleaseStateStore(tmp_path / "threshold-state.json")
    first = ThresholdReleaseCheckpoint(
        policy.policy_id,
        "stable",
        4,
        "sha256:" + "34" * 32,
    )
    assert store.record(first) == first
    assert store.record(first) == first
    assert store.floor(policy.policy_id, "stable") == first
    with pytest.raises(NativeBundleThresholdRollbackError, match="rollback"):
        store.record(
            ThresholdReleaseCheckpoint(
                policy.policy_id,
                "stable",
                3,
                "sha256:" + "56" * 32,
            )
        )
    with pytest.raises(NativeBundleThresholdError, match="same sequence"):
        store.record(
            ThresholdReleaseCheckpoint(
                policy.policy_id,
                "stable",
                4,
                "sha256:" + "78" * 32,
            )
        )


def test_threshold_channel_publish_fetch_load_and_rollback_replay(tmp_path: Path, monkeypatch) -> None:
    from tiny_tensor_compiler import native_bundle
    from tiny_tensor_compiler.native_bundle_threshold import (
        NativeBundleThresholdRollbackError,
        ThresholdReleaseStateStore,
        fetch_threshold_release_channel_archive,
        load_threshold_release_channel_registry,
        publish_threshold_release_channel,
    )

    policy = _policy()
    signers = (_key(1), _key(11))
    older = _archive(tmp_path, "threshold-older", (2,))
    newer = _archive(tmp_path, "threshold-newer", (2, 5))
    store = ThresholdReleaseStateStore(tmp_path / "trusted-threshold-state.json")

    with _server("secret") as (registry, state):
        release1 = publish_threshold_release_channel(
            older,
            registry,
            policy,
            signers,
            "stable",
            1,
            token="secret",
            allow_insecure_http=True,
        )
        channel_path = next(
            path for path in state.objects if path.startswith("/v1/channels/threshold-ed25519/")
        )
        replay = state.objects[channel_path]
        release2 = publish_threshold_release_channel(
            newer,
            registry,
            policy,
            signers,
            "stable",
            2,
            token="secret",
            allow_insecure_http=True,
        )
        assert release2.sequence == 2
        assert release2.archive_digest != release1.archive_digest

        destination = tmp_path / "threshold-current.ttca"
        accepted = fetch_threshold_release_channel_archive(
            registry,
            policy,
            "stable",
            destination,
            store,
            token="secret",
            allow_insecure_http=True,
        )
        assert accepted == release2
        assert destination.read_bytes() == newer.read_bytes()

        monkeypatch.setattr(
            native_bundle,
            "_compiler_command",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
        )
        executable = load_threshold_release_channel_registry(
            registry,
            policy,
            "stable",
            store,
            token="secret",
            allow_insecure_http=True,
        )
        try:
            assert executable.policy_id == policy.policy_id
            assert executable.sequence == 2
            x = np.arange(6, dtype=np.float32).reshape(2, 3) - np.float32(2)
            result = executable(inputs=[x])
            np.testing.assert_array_equal(result[0], np.maximum(x, np.float32(0)))
            np.testing.assert_array_equal(result[1], x + x)
        finally:
            executable.close()

        state.channel_substitute = replay
        before = len(state.requests)
        with pytest.raises(NativeBundleThresholdRollbackError, match="rollback"):
            fetch_threshold_release_channel_archive(
                registry,
                policy,
                "stable",
                tmp_path / "threshold-replayed.ttca",
                store,
                token="secret",
                allow_insecure_http=True,
            )
        replay_requests = state.requests[before:]
        assert any(
            path.startswith("/v1/channels/threshold-ed25519/")
            for _method, path, _auth in replay_requests
        )
        assert not any(
            path.startswith("/v1/archives/") for _method, path, _auth in replay_requests
        )


def test_threshold_publisher_rejects_non_monotonic_update(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_threshold import (
        NativeBundleThresholdError,
        publish_threshold_release_channel,
    )

    policy = _policy()
    archive = _archive(tmp_path, "threshold-release", (2,))
    with _server() as (registry, _state):
        publish_threshold_release_channel(
            archive,
            registry,
            policy,
            (_key(1), _key(11)),
            "stable",
            4,
            allow_insecure_http=True,
        )
        with pytest.raises(NativeBundleThresholdError, match="backward"):
            publish_threshold_release_channel(
                archive,
                registry,
                policy,
                (_key(1), _key(11)),
                "stable",
                3,
                allow_insecure_http=True,
            )

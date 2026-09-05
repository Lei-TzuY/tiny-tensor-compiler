from __future__ import annotations

import hashlib
import multiprocessing
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
            immutable = self.path.startswith(archive_prefix) or self.path.startswith(
                "/v1/attestations/ed25519/"
            )
            channel = self.path.startswith("/v1/channels/ed25519/")
            if self.path.startswith(archive_prefix):
                expected = self.path.removeprefix(archive_prefix)
                if hashlib.sha256(body).hexdigest() != expected:
                    self.send_error(400)
                    return
            elif not immutable and not channel:
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
                "/v1/channels/ed25519/"
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


def _record_state(path: str, sequence: int, digest: str, start) -> None:
    from tiny_tensor_compiler import NativeBundleRollbackError, ReleaseCheckpoint, ReleaseStateStore

    checkpoint = ReleaseCheckpoint(
        publisher_id="ed25519:" + "11" * 32,
        channel="stable",
        sequence=sequence,
        archive_digest=digest,
    )
    start.wait()
    try:
        ReleaseStateStore(path).record(checkpoint)
    except NativeBundleRollbackError:
        pass


def test_release_checkpoint_is_canonical_signed_and_domain_bound() -> None:
    from tiny_tensor_compiler import (
        PublisherTrustPolicy,
        create_release_checkpoint,
        publisher_public_key_from_private_key,
        verify_release_checkpoint,
    )

    secret = _key(0)
    policy = PublisherTrustPolicy((publisher_public_key_from_private_key(secret),))
    digest = "sha256:" + "ab" * 32
    encoded = create_release_checkpoint(secret, "stable", 7, digest)
    checkpoint = verify_release_checkpoint(encoded, policy, expected_channel="stable")
    assert checkpoint.channel == "stable"
    assert checkpoint.sequence == 7
    assert checkpoint.archive_digest == digest
    assert encoded.endswith(b"\n")

    tampered = encoded.replace(b'"sequence":7', b'"sequence":6')
    with pytest.raises(Exception, match="signature"):
        verify_release_checkpoint(tampered, policy, expected_channel="stable")
    with pytest.raises(Exception, match="channel"):
        verify_release_checkpoint(encoded, policy, expected_channel="beta")


def test_release_state_rejects_rollback_and_same_sequence_equivocation(tmp_path: Path) -> None:
    from tiny_tensor_compiler import (
        NativeBundleReleaseError,
        NativeBundleRollbackError,
        ReleaseCheckpoint,
        ReleaseStateStore,
    )

    store = ReleaseStateStore(tmp_path / "release-state.json")
    publisher = "ed25519:" + "22" * 32
    first = ReleaseCheckpoint(publisher, "stable", 4, "sha256:" + "aa" * 32)
    assert store.record(first) == first
    assert store.record(first) == first
    assert store.floor(publisher, "stable") == first

    with pytest.raises(NativeBundleRollbackError, match="rollback"):
        store.record(ReleaseCheckpoint(publisher, "stable", 3, "sha256:" + "bb" * 32))
    with pytest.raises(NativeBundleReleaseError, match="same sequence"):
        store.record(ReleaseCheckpoint(publisher, "stable", 4, "sha256:" + "cc" * 32))

    newer = ReleaseCheckpoint(publisher, "stable", 5, "sha256:" + "dd" * 32)
    assert store.record(newer) == newer
    assert store.floor(publisher, "stable") == newer


def test_release_state_cross_process_updates_never_lower_the_floor(tmp_path: Path) -> None:
    from tiny_tensor_compiler import ReleaseStateStore

    path = tmp_path / "release-state.json"
    digest10 = "sha256:" + "10" * 32
    digest11 = "sha256:" + "11" * 32
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    workers = [
        ctx.Process(target=_record_state, args=(str(path), 10, digest10, start)),
        ctx.Process(target=_record_state, args=(str(path), 11, digest11, start)),
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    floor = ReleaseStateStore(path).floor("ed25519:" + "11" * 32, "stable")
    assert floor is not None
    assert floor.sequence == 11
    assert floor.archive_digest == digest11


def test_release_channel_publish_fetch_load_and_rollback_replay(tmp_path: Path, monkeypatch) -> None:
    from tiny_tensor_compiler import (
        NativeBundleRollbackError,
        PublisherTrustPolicy,
        ReleaseStateStore,
        fetch_release_channel_archive,
        load_release_channel_registry,
        publish_release_channel,
        publisher_public_key_from_private_key,
    )
    from tiny_tensor_compiler import native_bundle

    secret = _key(3)
    policy = PublisherTrustPolicy((publisher_public_key_from_private_key(secret),))
    older = _archive(tmp_path, "older", (2,))
    newer = _archive(tmp_path, "newer", (2, 5))
    store = ReleaseStateStore(tmp_path / "trusted-release-state.json")

    with _server("secret") as (registry, state):
        release1 = publish_release_channel(
            older,
            registry,
            secret,
            "stable",
            1,
            token="secret",
            allow_insecure_http=True,
        )
        channel_path = next(path for path in state.objects if path.startswith("/v1/channels/"))
        replay = state.objects[channel_path]
        release2 = publish_release_channel(
            newer,
            registry,
            secret,
            "stable",
            2,
            token="secret",
            allow_insecure_http=True,
        )
        assert release2.sequence == 2
        assert release2.archive_digest != release1.archive_digest

        fetched = tmp_path / "current.ttca"
        accepted = fetch_release_channel_archive(
            registry,
            release2.publisher_id,
            "stable",
            fetched,
            policy,
            store,
            token="secret",
            allow_insecure_http=True,
        )
        assert accepted == release2
        assert fetched.read_bytes() == newer.read_bytes()

        monkeypatch.setattr(
            native_bundle,
            "_compiler_command",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
        )
        executable = load_release_channel_registry(
            registry,
            release2.publisher_id,
            "stable",
            policy,
            store,
            token="secret",
            allow_insecure_http=True,
        )
        try:
            assert executable.sequence == 2
            assert executable.channel == "stable"
            x = np.arange(6, dtype=np.float32).reshape(2, 3) - np.float32(2)
            result = executable(inputs=[x])
            np.testing.assert_array_equal(result[0], np.maximum(x, np.float32(0)))
            np.testing.assert_array_equal(result[1], x + x)
        finally:
            executable.close()

        state.channel_substitute = replay
        before = len(state.requests)
        with pytest.raises(NativeBundleRollbackError, match="rollback"):
            fetch_release_channel_archive(
                registry,
                release2.publisher_id,
                "stable",
                tmp_path / "replayed.ttca",
                policy,
                store,
                token="secret",
                allow_insecure_http=True,
            )
        replay_requests = state.requests[before:]
        assert any(path.startswith("/v1/channels/") for _method, path, _auth in replay_requests)
        assert not any(path.startswith("/v1/archives/") for _method, path, _auth in replay_requests)


def test_release_publisher_rejects_non_monotonic_update(tmp_path: Path) -> None:
    from tiny_tensor_compiler import NativeBundleReleaseError, publish_release_channel

    secret = _key(7)
    archive = _archive(tmp_path, "release", (2,))
    with _server() as (registry, _state):
        publish_release_channel(archive, registry, secret, "stable", 4, allow_insecure_http=True)
        with pytest.raises(NativeBundleReleaseError, match="advance"):
            publish_release_channel(
                archive,
                registry,
                secret,
                "stable",
                3,
                allow_insecure_http=True,
            )

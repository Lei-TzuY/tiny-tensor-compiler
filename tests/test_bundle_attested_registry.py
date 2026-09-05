from __future__ import annotations

import hashlib
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
        self.requests: list[tuple[str, str, str | None]] = []
        self.attestation_substitute: bytes | None = None


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
            if self.path.startswith(archive_prefix):
                expected = self.path.removeprefix(archive_prefix)
                if hashlib.sha256(body).hexdigest() != expected:
                    self.send_error(400)
                    return
            elif not self.path.startswith("/v1/attestations/ed25519/"):
                self.send_error(404)
                return
            if self.path in state.objects:
                self.send_response(412)
                self.end_headers()
                return
            state.objects[self.path] = body
            self.send_response(201)
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
            if (
                state.attestation_substitute is not None
                and self.path.startswith("/v1/attestations/ed25519/")
            ):
                body = state.attestation_substitute
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
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


def _archive(tmp_path: Path) -> Path:
    from tiny_tensor_compiler.native_bundle_archive import pack_dynamic_bundle_set_archive
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 3), dtype="float32")
    module = builder.finish((x.relu(), x + x))
    bundle = tmp_path / "family.ttcset"
    archive = tmp_path / "family.ttca"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    pack_dynamic_bundle_set_archive(bundle, archive)
    return archive


def _input(batch: int) -> np.ndarray:
    return np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) - np.float32(2)


def _assert_outputs(result, x: np.ndarray) -> None:
    np.testing.assert_array_equal(result[0], np.maximum(x, np.float32(0)))
    np.testing.assert_array_equal(result[1], x + x)


def test_attested_publish_fetch_and_compiler_free_load(tmp_path: Path, monkeypatch) -> None:
    from tiny_tensor_compiler import (
        PublisherTrustPolicy,
        fetch_attested_dynamic_bundle_set_archive,
        load_attested_dynamic_bundle_set_registry,
        publish_attested_dynamic_bundle_set_archive,
        publisher_public_key_from_private_key,
    )
    from tiny_tensor_compiler import native_bundle

    archive = _archive(tmp_path)
    secret = _key(0)
    public = publisher_public_key_from_private_key(secret)
    policy = PublisherTrustPolicy((public,))
    with _server("secret") as (registry, state):
        digest, publisher = publish_attested_dynamic_bundle_set_archive(
            archive,
            registry,
            secret,
            token="secret",
            allow_insecure_http=True,
        )
        assert publish_attested_dynamic_bundle_set_archive(
            archive,
            registry,
            secret,
            token="secret",
            allow_insecure_http=True,
        ) == (digest, publisher)

        fetched = tmp_path / "trusted.ttca"
        fetch_attested_dynamic_bundle_set_archive(
            registry,
            digest,
            publisher,
            fetched,
            policy,
            token="secret",
            allow_insecure_http=True,
        )
        assert fetched.read_bytes() == archive.read_bytes()

        monkeypatch.setattr(
            native_bundle,
            "_compiler_command",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
        )
        executable = load_attested_dynamic_bundle_set_registry(
            registry,
            digest,
            publisher,
            policy,
            token="secret",
            allow_insecure_http=True,
        )
        try:
            assert executable.digest == digest
            assert executable.publisher_id == publisher
            x = _input(2)
            _assert_outputs(executable(inputs=[x]), x)
        finally:
            executable.close()
        assert executable.closed
        assert all(auth == "Bearer secret" for _method, _path, auth in state.requests)


def test_attested_fetch_rejects_replaced_attestation_before_destination_publish(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler import (
        NativeBundleTrustError,
        PublisherTrustPolicy,
        create_archive_attestation,
        fetch_attested_dynamic_bundle_set_archive,
        publish_attested_dynamic_bundle_set_archive,
        publisher_public_key_from_private_key,
    )

    archive = _archive(tmp_path)
    secret = _key(0)
    policy = PublisherTrustPolicy((publisher_public_key_from_private_key(secret),))
    with _server() as (registry, state):
        digest, publisher = publish_attested_dynamic_bundle_set_archive(
            archive, registry, secret, allow_insecure_http=True
        )
        state.attestation_substitute = create_archive_attestation(_key(11), digest)
        destination = tmp_path / "rejected.ttca"
        with pytest.raises(NativeBundleTrustError, match="identity mismatch"):
            fetch_attested_dynamic_bundle_set_archive(
                registry,
                digest,
                publisher,
                destination,
                policy,
                allow_insecure_http=True,
            )
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".rejected.ttca.attested-*"))


def test_revoked_publisher_is_rejected_before_network_access(tmp_path: Path) -> None:
    from tiny_tensor_compiler import (
        NativeBundleTrustError,
        PublisherTrustPolicy,
        fetch_attested_dynamic_bundle_set_archive,
        publish_attested_dynamic_bundle_set_archive,
        publisher_public_key_from_private_key,
    )

    archive = _archive(tmp_path)
    secret = _key(0)
    public = publisher_public_key_from_private_key(secret)
    with _server() as (registry, state):
        digest, publisher = publish_attested_dynamic_bundle_set_archive(
            archive, registry, secret, allow_insecure_http=True
        )
        before = len(state.requests)
        policy = PublisherTrustPolicy(
            (public,), revoked_publishers=frozenset({publisher})
        )
        with pytest.raises(NativeBundleTrustError, match="revoked"):
            fetch_attested_dynamic_bundle_set_archive(
                registry,
                digest,
                publisher,
                tmp_path / "revoked.ttca",
                policy,
                allow_insecure_http=True,
            )
        assert len(state.requests) == before

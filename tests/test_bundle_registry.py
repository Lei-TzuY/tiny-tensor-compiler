from __future__ import annotations

import hashlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim


class _RegistryState:
    def __init__(self, *, token: str | None = None) -> None:
        self.token = token
        self.objects: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.redirect_url: str | None = None
        self.substitute: bytes | None = None
        self.truncate = False


def _handler_for(state: _RegistryState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format, *_args) -> None:
            return

        def _record(self) -> None:
            state.requests.append(
                (self.command, self.path, self.headers.get("Authorization"))
            )

        def _authorized(self) -> bool:
            if state.token is None:
                return True
            return self.headers.get("Authorization") == f"Bearer {state.token}"

        def _digest_from_path(self) -> str | None:
            prefix = "/v1/archives/sha256/"
            if not self.path.startswith(prefix):
                return None
            digest = self.path.removeprefix(prefix)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                return None
            return digest

        def do_PUT(self) -> None:
            self._record()
            if not self._authorized():
                self.send_error(401)
                return
            digest = self._digest_from_path()
            if digest is None:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0:
                self.send_error(411)
                return
            body = self.rfile.read(length)
            if hashlib.sha256(body).hexdigest() != digest:
                self.send_error(400)
                return
            if digest in state.objects:
                self.send_response(412)
                self.end_headers()
                return
            state.objects[digest] = body
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            self._record()
            if state.redirect_url is not None:
                self.send_response(302)
                self.send_header("Location", state.redirect_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if not self._authorized():
                self.send_error(401)
                return
            digest = self._digest_from_path()
            if digest is None or digest not in state.objects:
                self.send_error(404)
                return
            body = state.objects[digest] if state.substitute is None else state.substitute
            self.send_response(200)
            self.send_header(
                "Content-Type", "application/vnd.tiny-tensor-compiler.bundle-archive"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.truncate:
                self.wfile.write(body[: max(1, len(body) // 2)])
            else:
                self.wfile.write(body)

    return Handler


@contextmanager
def _registry_server(*, token: str | None = None):
    state = _RegistryState(token=token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _dynamic_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 3), dtype="float32")
    module = builder.finish((x.relu(), x + x))
    return module, batch


def _input(batch: int) -> np.ndarray:
    values = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
    return values - np.float32(2)


def _assert_outputs(result, x: np.ndarray) -> None:
    relu, added = result
    np.testing.assert_array_equal(relu, np.maximum(x, np.float32(0)))
    np.testing.assert_array_equal(added, x + x)


def _compile_archive(tmp_path: Path) -> Path:
    from tiny_tensor_compiler.native_bundle_archive import pack_dynamic_bundle_set_archive
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    archive = tmp_path / "family.ttca"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    pack_dynamic_bundle_set_archive(bundle, archive)
    return archive


def test_registry_publish_fetch_and_load_are_content_addressed_and_compiler_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle
    from tiny_tensor_compiler.native_bundle_registry import (
        fetch_dynamic_bundle_set_archive,
        load_dynamic_bundle_set_registry,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server(token="secret") as (registry, state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            token="secret",
            allow_insecure_http=True,
        )
        assert digest == f"sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}"
        assert state.objects[digest.removeprefix("sha256:")] == archive.read_bytes()

        # Immutable content-addressed publication is idempotent only after the client
        # verifies the already-existing object through a full GET/archive validation.
        assert (
            publish_dynamic_bundle_set_archive(
                archive,
                registry,
                token="secret",
                allow_insecure_http=True,
            )
            == digest
        )

        fetched = tmp_path / "downloaded.ttca"
        fetch_dynamic_bundle_set_archive(
            registry,
            digest,
            fetched,
            token="secret",
            allow_insecure_http=True,
        )
        assert fetched.read_bytes() == archive.read_bytes()

        monkeypatch.setattr(
            native_bundle,
            "_compiler_command",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
        )
        executable = load_dynamic_bundle_set_registry(
            registry,
            digest,
            token="secret",
            allow_insecure_http=True,
        )
        try:
            assert executable.digest == digest
            assert executable.available_bindings == ((("B", 2),), (("B", 5),))
            x2 = _input(2)
            _assert_outputs(executable(inputs=[x2]), x2)
            child = executable.specialize({"B": 5})
            x5 = _input(5)
            out0 = np.empty_like(x5)
            out1 = np.empty_like(x5)
            result = child(inputs=[x5], out=(out0, out1))
            assert result[0] is out0
            assert result[1] is out1
            _assert_outputs(result, x5)
        finally:
            executable.close()
        assert executable.closed

        assert any(method == "PUT" for method, _path, _auth in state.requests)
        assert all(
            auth == "Bearer secret"
            for _method, _path, auth in state.requests
        )


def test_registry_requires_explicit_opt_in_for_plain_http(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import publish_dynamic_bundle_set_archive

    archive = _compile_archive(tmp_path)
    with _registry_server() as (registry, _state):
        with pytest.raises(ValueError, match="allow_insecure_http=True"):
            publish_dynamic_bundle_set_archive(archive, registry)


def test_registry_fetch_rejects_server_substitution_and_cleans_destination(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        fetch_dynamic_bundle_set_archive,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server() as (registry, state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            allow_insecure_http=True,
        )
        state.substitute = archive.read_bytes() + b"replacement"
        destination = tmp_path / "substituted.ttca"
        with pytest.raises(NativeBundleRegistryError, match="digest mismatch"):
            fetch_dynamic_bundle_set_archive(
                registry,
                digest,
                destination,
                allow_insecure_http=True,
            )
        assert not destination.exists()
        assert not tuple(tmp_path.glob(".substituted.ttca.download-*.tmp"))


def test_registry_publish_does_not_trust_successful_upload_response(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server() as (registry, state):
        state.substitute = b"coherent server substitution is still not the pinned object"
        with pytest.raises(NativeBundleRegistryError, match="post-publication verification"):
            publish_dynamic_bundle_set_archive(
                archive,
                registry,
                allow_insecure_http=True,
            )


def test_registry_fetch_enforces_transfer_limit_before_publication(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        fetch_dynamic_bundle_set_archive,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server() as (registry, _state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            allow_insecure_http=True,
        )
        destination = tmp_path / "oversize.ttca"
        with pytest.raises(NativeBundleRegistryError, match="transfer limit"):
            fetch_dynamic_bundle_set_archive(
                registry,
                digest,
                destination,
                allow_insecure_http=True,
                max_bytes=max(1, archive.stat().st_size - 1),
            )
        assert not destination.exists()


def test_registry_refuses_redirect_without_forwarding_bearer_token(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        fetch_dynamic_bundle_set_archive,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server(token="secret") as (registry, state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            token="secret",
            allow_insecure_http=True,
        )
        with _registry_server() as (other, other_state):
            state.redirect_url = f"{other}/stolen"
            with pytest.raises(NativeBundleRegistryError, match="HTTP status 302"):
                fetch_dynamic_bundle_set_archive(
                    registry,
                    digest,
                    tmp_path / "redirect.ttca",
                    token="secret",
                    allow_insecure_http=True,
                )
            assert other_state.requests == []


def test_registry_rejects_wrong_credentials_and_missing_objects(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        fetch_dynamic_bundle_set_archive,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server(token="secret") as (registry, _state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            token="secret",
            allow_insecure_http=True,
        )
        with pytest.raises(NativeBundleRegistryError, match="HTTP status 401"):
            fetch_dynamic_bundle_set_archive(
                registry,
                digest,
                tmp_path / "unauthorized.ttca",
                token="wrong",
                allow_insecure_http=True,
            )
        missing = "sha256:" + "0" * 64
        with pytest.raises(NativeBundleRegistryError, match="HTTP status 404"):
            fetch_dynamic_bundle_set_archive(
                registry,
                missing,
                tmp_path / "missing.ttca",
                token="secret",
                allow_insecure_http=True,
            )


def test_registry_rejects_truncated_response(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import (
        NativeBundleRegistryError,
        fetch_dynamic_bundle_set_archive,
        publish_dynamic_bundle_set_archive,
    )

    archive = _compile_archive(tmp_path)
    with _registry_server() as (registry, state):
        digest = publish_dynamic_bundle_set_archive(
            archive,
            registry,
            allow_insecure_http=True,
        )
        state.truncate = True
        with pytest.raises((NativeBundleRegistryError, OSError)):
            fetch_dynamic_bundle_set_archive(
                registry,
                digest,
                tmp_path / "truncated.ttca",
                allow_insecure_http=True,
            )


def test_registry_validates_canonical_digest_and_destination_collision(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_registry import fetch_dynamic_bundle_set_archive

    with pytest.raises(ValueError, match="canonical sha256"):
        fetch_dynamic_bundle_set_archive(
            "https://registry.example",
            "SHA256:" + "A" * 64,
            tmp_path / "bad.ttca",
        )

    destination = tmp_path / "existing.ttca"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        fetch_dynamic_bundle_set_archive(
            "https://registry.example",
            "sha256:" + "0" * 64,
            destination,
        )

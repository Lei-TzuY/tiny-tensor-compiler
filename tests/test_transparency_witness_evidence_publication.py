from __future__ import annotations

import hashlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tiny_tensor_compiler import (
    NativeBundleTransparencyRollbackError,
    TransparencyStateStore,
    TransparencyWitnessPolicy,
    create_release_checkpoint,
    create_transparency_checkpoint,
    publisher_public_key_from_private_key,
    verify_transparency_checkpoint,
)
from tiny_tensor_compiler.native_bundle_transparency_witness import (
    NativeBundleTransparencyWitnessError,
)
from tiny_tensor_compiler.native_bundle_transparency_witness_evidence import (
    TransparencyWitnessEvidenceStore,
)
from tiny_tensor_compiler.native_bundle_transparency_witness_evidence_publication import (
    NativeBundleTransparencyPublicationError,
    accept_transparency_witness_evidence_publication,
    create_transparency_witness_evidence_publication,
    fetch_transparency_witness_evidence_publication,
    verify_transparency_witness_evidence_publication,
)
from tiny_tensor_compiler.native_bundle_transparency_witness_observation import (
    create_transparency_witness_observation,
    verify_transparency_witness_observation,
)

_MEDIA_TYPE = "application/vnd.tiny-tensor-compiler.transparency-witness-publication+json"


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


class _PublicationState:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.content_type = _MEDIA_TYPE
        self.redirect_url: str | None = None
        self.truncate = False
        self.requests = 0


def _handler_for(state: _PublicationState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format, *_args) -> None:
            return

        def do_GET(self) -> None:
            state.requests += 1
            if state.redirect_url is not None:
                self.send_response(302)
                self.send_header("Location", state.redirect_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path != "/publication":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", state.content_type)
            self.send_header("Content-Length", str(len(state.body)))
            self.end_headers()
            if state.truncate:
                self.wfile.write(state.body[: max(1, len(state.body) // 2)])
            else:
                self.wfile.write(state.body)

    return Handler


@contextmanager
def _publication_server(body: bytes):
    state = _PublicationState(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/publication", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_publication_roundtrip_binds_publisher_log_policy_observation_and_proofs(
    tmp_path: Path,
) -> None:
    log_private = _key(120)
    log_public = _public(120)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3)]
    old = _observation(
        tmp_path,
        state_name="old.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    new = _observation(
        tmp_path,
        state_name="new.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=leaves,
    )
    old_digest = _digest(old, log_public=log_public, policy=policy)
    proof = _consistency_proof(1, leaves)

    encoded = create_transparency_witness_evidence_publication(
        _key(90),
        new,
        log_public_key=log_public,
        policy=policy,
        consistency_proofs={old_digest: proof},
    )
    verified = verify_transparency_witness_evidence_publication(
        encoded,
        publisher_public_key=_public(90),
        log_public_key=log_public,
        policy=policy,
    )

    assert verified.encoded_publication == encoded
    assert verified.publisher_id
    assert verified.log_id == verify_transparency_witness_observation(
        new, log_public_key=log_public, policy=policy
    ).checkpoint.log_id
    assert verified.policy_id == policy.policy_id
    assert verified.observation.encoded_observation == new
    assert dict(verified.consistency_proofs) == {old_digest: proof}

    with pytest.raises(NativeBundleTransparencyPublicationError, match="publisher identity"):
        verify_transparency_witness_evidence_publication(
            encoded,
            publisher_public_key=_public(91),
            log_public_key=log_public,
            policy=policy,
        )

    tampered = bytearray(encoded)
    tampered[-10] ^= 1
    with pytest.raises(NativeBundleTransparencyPublicationError):
        verify_transparency_witness_evidence_publication(
            bytes(tampered),
            publisher_public_key=_public(90),
            log_public_key=log_public,
            policy=policy,
        )


def test_real_http_fetch_and_accept_advances_durable_client_state(tmp_path: Path) -> None:
    log_private = _key(121)
    log_public = _public(121)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3, 4)]
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
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(old)
    encoded = create_transparency_witness_evidence_publication(
        _key(92),
        new,
        log_public_key=log_public,
        policy=policy,
        consistency_proofs={old_digest: _consistency_proof(1, leaves)},
    )

    with _publication_server(encoded) as (url, state):
        fetched = fetch_transparency_witness_evidence_publication(
            url,
            allow_insecure_http=True,
        )
        snapshot = accept_transparency_witness_evidence_publication(
            fetched,
            publisher_public_key=_public(92),
            log_public_key=log_public,
            policy=policy,
            evidence_store=store,
        )

    assert state.requests == 1
    assert snapshot.status == "healthy"
    assert sorted(item.checkpoint.tree_size for item in snapshot.observations) == [1, 4]
    assert TransparencyWitnessEvidenceStore(
        tmp_path / "evidence.json", log_public, policy
    ).current() == snapshot


def test_http_transport_requires_explicit_insecure_opt_in_and_refuses_redirects(
    tmp_path: Path,
) -> None:
    log_private = _key(122)
    log_public = _public(122)
    policy = TransparencyWitnessPolicy((_public(1),), threshold=1)
    observation = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    encoded = create_transparency_witness_evidence_publication(
        _key(93),
        observation,
        log_public_key=log_public,
        policy=policy,
    )

    with _publication_server(encoded) as (url, state):
        with pytest.raises(ValueError, match="allow_insecure_http=True"):
            fetch_transparency_witness_evidence_publication(url)
        state.redirect_url = url
        with pytest.raises(NativeBundleTransparencyPublicationError, match="HTTP status 302"):
            fetch_transparency_witness_evidence_publication(
                url,
                allow_insecure_http=True,
            )


def test_http_transport_rejects_wrong_media_type_truncation_and_size_limit(tmp_path: Path) -> None:
    log_private = _key(123)
    log_public = _public(123)
    policy = TransparencyWitnessPolicy((_public(1),), threshold=1)
    observation = _observation(
        tmp_path,
        state_name="w1.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    encoded = create_transparency_witness_evidence_publication(
        _key(94),
        observation,
        log_public_key=log_public,
        policy=policy,
    )

    with _publication_server(encoded) as (url, state):
        state.content_type = "application/octet-stream"
        with pytest.raises(NativeBundleTransparencyPublicationError, match="media type"):
            fetch_transparency_witness_evidence_publication(
                url,
                allow_insecure_http=True,
            )
        state.content_type = _MEDIA_TYPE
        state.truncate = True
        with pytest.raises(NativeBundleTransparencyPublicationError, match="Content-Length"):
            fetch_transparency_witness_evidence_publication(
                url,
                allow_insecure_http=True,
            )
        state.truncate = False
        with pytest.raises(NativeBundleTransparencyPublicationError, match="transfer limit"):
            fetch_transparency_witness_evidence_publication(
                url,
                allow_insecure_http=True,
                max_bytes=len(encoded) - 1,
            )


def test_accept_rechecks_store_required_proofs_before_mutation(tmp_path: Path) -> None:
    log_private = _key(124)
    log_public = _public(124)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3)]
    old = _observation(
        tmp_path,
        state_name="old.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    new = _observation(
        tmp_path,
        state_name="new.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=leaves,
    )
    path = tmp_path / "evidence.json"
    store = TransparencyWitnessEvidenceStore(path, log_public, policy)
    baseline = store.record(old)

    missing = create_transparency_witness_evidence_publication(
        _key(95),
        new,
        log_public_key=log_public,
        policy=policy,
    )
    with pytest.raises(NativeBundleTransparencyWitnessError, match="consistency proof set"):
        accept_transparency_witness_evidence_publication(
            missing,
            publisher_public_key=_public(95),
            log_public_key=log_public,
            policy=policy,
            evidence_store=store,
        )
    assert store.current() == baseline

    old_digest = _digest(old, log_public=log_public, policy=policy)
    proof = _consistency_proof(1, leaves)
    damaged = (*proof[:-1], b"\xff" * 32)
    bad = create_transparency_witness_evidence_publication(
        _key(95),
        new,
        log_public_key=log_public,
        policy=policy,
        consistency_proofs={old_digest: damaged},
    )
    with pytest.raises(Exception, match="consistency proof"):
        accept_transparency_witness_evidence_publication(
            bad,
            publisher_public_key=_public(95),
            log_public_key=log_public,
            policy=policy,
            evidence_store=store,
        )
    assert store.current() == baseline


def test_fetched_same_size_fork_becomes_terminal_signed_evidence(tmp_path: Path) -> None:
    log_private = _key(125)
    log_public = _public(125)
    policy = TransparencyWitnessPolicy((_public(1), _public(2)), threshold=1)
    first = _observation(
        tmp_path,
        state_name="first.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=[_release(1)],
    )
    conflicting = _observation(
        tmp_path,
        state_name="fork.json",
        witness_seed=2,
        log_private=log_private,
        policy=policy,
        leaves=[_release(99)],
    )
    store = TransparencyWitnessEvidenceStore(tmp_path / "evidence.json", log_public, policy)
    store.record(first)
    encoded = create_transparency_witness_evidence_publication(
        _key(96),
        conflicting,
        log_public_key=log_public,
        policy=policy,
    )

    with _publication_server(encoded) as (url, _state):
        fetched = fetch_transparency_witness_evidence_publication(
            url,
            allow_insecure_http=True,
        )
    forked = accept_transparency_witness_evidence_publication(
        fetched,
        publisher_public_key=_public(96),
        log_public_key=log_public,
        policy=policy,
        evidence_store=store,
    )

    assert forked.status == "forked"
    assert forked.fork_evidence is not None
    assert forked.fork_evidence[0].checkpoint.tree_size == 1
    assert forked.fork_evidence[1].checkpoint.tree_size == 1
    with pytest.raises(NativeBundleTransparencyWitnessError, match="terminal fork"):
        store.record(first)


def test_signed_old_publication_cannot_roll_back_newer_client_state(tmp_path: Path) -> None:
    log_private = _key(126)
    log_public = _public(126)
    policy = TransparencyWitnessPolicy((_public(1),), threshold=1)
    leaves = [_release(index) for index in (1, 2, 3)]
    old = _observation(
        tmp_path,
        state_name="old.json",
        witness_seed=1,
        log_private=log_private,
        policy=policy,
        leaves=leaves[:1],
    )
    new = _observation(
        tmp_path,
        state_name="new.json",
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
    encoded_old = create_transparency_witness_evidence_publication(
        _key(97),
        old,
        log_public_key=log_public,
        policy=policy,
    )

    with pytest.raises(NativeBundleTransparencyRollbackError, match="rollback"):
        accept_transparency_witness_evidence_publication(
            encoded_old,
            publisher_public_key=_public(97),
            log_public_key=log_public,
            policy=policy,
            evidence_store=store,
        )
    assert store.current() == advanced

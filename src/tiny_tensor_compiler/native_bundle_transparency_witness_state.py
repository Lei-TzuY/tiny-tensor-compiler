from __future__ import annotations

from collections.abc import Sequence

from .native_bundle_attestation import (
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_transparency import (
    NativeBundleTransparencyError,
    TransparencyStateStore,
    verify_transparency_checkpoint,
)
from .native_bundle_transparency_witness import (
    NativeBundleTransparencyWitnessError,
    TransparencyWitnessPolicy,
    create_transparency_witness_quorum,
)


def create_stateful_transparency_witness_quorum(
    private_keys: Sequence[bytes],
    policy: TransparencyWitnessPolicy,
    encoded_checkpoint: bytes,
    *,
    log_public_key: bytes,
    state_stores: Sequence[TransparencyStateStore],
    consistency_proofs: Sequence[Sequence[bytes]],
) -> bytes:
    """Create a quorum only after every selected witness accepts log consistency.

    Each signer owns a distinct persistent ``TransparencyStateStore`` for the same
    pinned log. All stores precheck the checkpoint before any store is advanced.
    Each ``record`` call then rechecks under its own state lock. The existing quorum
    encoder signs only after every selected witness has persisted the checkpoint.

    First contact retains the base transparency model's limitation: an empty witness
    state has no independent freshness information.
    """
    if not isinstance(policy, TransparencyWitnessPolicy):
        raise TypeError("policy must be a TransparencyWitnessPolicy")

    keys = tuple(private_keys)
    stores = tuple(state_stores)
    proofs = tuple(tuple(proof) for proof in consistency_proofs)
    if not keys:
        raise ValueError("stateful witness quorum requires at least one signer")
    if len(keys) != len(stores) or len(keys) != len(proofs):
        raise ValueError(
            "stateful witness keys, state stores, and consistency proofs must have equal length"
        )
    if len(keys) < policy.threshold:
        raise NativeBundleTransparencyWitnessError(
            f"transparency witness quorum requires at least {policy.threshold} distinct eligible signatures"
        )

    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public_key)

    witness_ids: list[str] = []
    for private_key in keys:
        public_key = publisher_public_key_from_private_key(private_key)
        witness_id = publisher_id_from_public_key(public_key)
        policy.public_key_for(witness_id)
        if witness_id in witness_ids:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness quorum contains a duplicate witness"
            )
        witness_ids.append(witness_id)

    paths = []
    for store in stores:
        if not isinstance(store, TransparencyStateStore):
            raise TypeError("state_stores must contain TransparencyStateStore values")
        if store.log_id != checkpoint.log_id:
            raise NativeBundleTransparencyError(
                "stateful witness state store uses a different pinned log operator"
            )
        paths.append(store.path)
    if len(set(paths)) != len(paths):
        raise ValueError("stateful witnesses must use distinct state store paths")

    # Fail deterministic bad input before mutating any witness state.
    for store, proof in zip(stores, proofs, strict=True):
        store.precheck(checkpoint, proof)

    # ``record`` repeats the consistency check while holding each store's state lock,
    # protecting against a concurrent state advance between precheck and persistence.
    for store, proof in zip(stores, proofs, strict=True):
        store.record(checkpoint, proof)

    # Reuse the established canonical quorum schema and signature domain. Clients do
    # not need a second verifier or wire format for stateful witnesses.
    return create_transparency_witness_quorum(keys, policy, encoded_checkpoint)

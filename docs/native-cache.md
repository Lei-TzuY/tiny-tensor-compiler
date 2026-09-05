# Persistent native cache integrity

The optional `cache_dir=` native cache is content-addressed by the exact generated C source, compiler command, compiler fingerprint, target fingerprint, and cache schema. The current `native-v2` schema authenticates the bytes of every cached shared library before that library can be staged or loaded, and coordinates concurrent builders with one operating-system-backed lease per cache digest.

Each persistent entry contains two files under `native-v2/<cache-digest>/`:

- the platform shared library (`program.so`, `program.dylib`, or `program.dll`);
- `manifest.json`, which records the schema, cache digest, library filename, and SHA-256 of the exact library bytes.

A separate `native-v2/.<cache-digest>.lock` file names the build lease for that digest. The file's existence is not ownership: the operating system lock on its open file descriptor is the lease. Different digests therefore use independent locks and may build concurrently.

Reuse follows a fail-closed sequence under the per-digest lease. The runtime requires both cache files, parses the manifest, verifies its schema/digest/filename, recomputes the shared-library SHA-256, and only then copies the library into a process-owned staging directory for dynamic loading. A missing or malformed manifest, metadata mismatch, hash mismatch, or a checksum-valid library that still fails to load invalidates the persistent entry and triggers recompilation.

A process that initially observes a cache miss does not compile immediately. It first acquires the digest lease and then re-runs the complete verified-entry check. If another process published a valid artifact while this process was waiting, the follower stages that artifact and skips compilation. This makes same-key publication deterministic and suppresses duplicate builds without weakening the existing integrity checks.

Compilation happens in a temporary build directory while the digest lease is held. The compiled library hash is recorded before publication; the library is published first and its manifest second, and the newly published pair is revalidated and staged before the lease is released. No cooperating process can validate, invalidate, or publish the same digest concurrently. If publication fails, the incomplete pair is invalidated while the lease is still held.

Lease lifetime is tied to the operating-system file lock, not to a PID file or a cleanup protocol. POSIX uses `flock`; Windows uses a one-byte `msvcrt.locking` lease with bounded polling. Closing the file descriptor releases the lease, and process termination releases it even when normal Python cleanup or a context-manager `finally` block does not run. The persistent lock file may remain on disk and be reused by later processes without representing stale ownership.

The persistent library files themselves are never executed in place. Valid entries are copied into process-owned staging directories, preserving the existing Windows and POSIX library-lifecycle behavior. `clear_native_cache()` releases only process-local staged artifacts; it does not remove validated persistent entries or cross-process lease files.

This layer is a correctness and durability contract, not a performance claim. It proves per-key mutual exclusion, independent progress for different digests, crash-safe lease release, and follower revalidation/reuse. It does not promise wall-clock speedup, compiler throughput, fairness between waiting processes, or exactly-once compilation across unrelated cache keys. The `native-v2` digest, manifest schema, native ABI, and process-local artifact cache remain unchanged.

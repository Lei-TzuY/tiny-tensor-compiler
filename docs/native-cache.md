# Persistent native cache integrity

The optional `cache_dir=` native cache is content-addressed by the exact generated C source, compiler command, compiler fingerprint, target fingerprint, and cache schema. The current `native-v2` schema also authenticates the bytes of every cached shared library before that library can be staged or loaded.

Each persistent entry contains two files under `native-v2/<cache-digest>/`:

- the platform shared library (`program.so`, `program.dylib`, or `program.dll`);
- `manifest.json`, which records the schema, cache digest, library filename, and SHA-256 of the exact library bytes.

Reuse follows a fail-closed sequence. The runtime first requires both files, parses the manifest, verifies its schema/digest/filename, recomputes the shared-library SHA-256, and only then copies the library into a process-owned staging directory for dynamic loading. A missing or malformed manifest, metadata mismatch, hash mismatch, or a checksum-valid library that still fails to load invalidates the persistent entry and triggers recompilation. This means a different but otherwise loadable shared library cannot be substituted under an existing cache key and executed accidentally.

Compilation happens in a temporary build directory. The compiled library hash is recorded before publication; the library is published first and its manifest second. A crash between those writes leaves an incomplete entry, which is rejected and rebuilt on the next use. Failed compilation therefore cannot publish a valid manifest or poison later executions.

The persistent files themselves are never executed in place. Valid entries are copied into process-owned staging directories, preserving the existing Windows and POSIX library-lifecycle behavior. `clear_native_cache()` releases only process-local staged artifacts; it does not remove validated persistent entries.

This integrity layer is a correctness and durability contract, not a performance claim. It does not implement a cross-process compilation lock or promise that concurrent processes will deduplicate compilation work. Concurrent/incomplete publication is handled by validation and recovery rather than by claiming exactly-once compilation.

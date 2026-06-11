# Releasing Minutist

## Version single source of truth

The release version lives in one place: `Cargo.toml` (the workspace root),
`[workspace.package] version = "..."`. The `src-tauri/Cargo.toml` inherits
it via `version.workspace = true`. `tauri.conf.json` is set to
`"version": null` so the Tauri bundler reads the Cargo crate version
directly — no second edit is needed.

## Release steps

1. **Bump the version** in `Cargo.toml` (workspace root), e.g. `0.1.0`:

   ```toml
   [workspace.package]
   version = "0.1.0"
   ```

   Run `cargo check` to refresh `Cargo.lock`.

2. **Commit the bump**:

   ```sh
   git add Cargo.toml Cargo.lock
   git commit -m "chore: bump version to 0.1.0"
   ```

3. **Push a `v*` tag** to trigger `release.yml`:

   ```sh
   git tag v0.1.0
   git push origin main v0.1.0
   ```

4. **GitHub Actions** (`release.yml`) builds signed release bundles for
   Linux (Vulkan), Windows (Vulkan), and macOS (Metal), then creates a
   draft GitHub Release named `minutist v0.1.0` with:
   - Per-platform signed installers and their `.sig` files.
   - `latest.json` — the `tauri-plugin-updater` manifest (produced by
     `includeUpdaterJson: true`).

5. **Review and publish** the draft Release on GitHub.

## What `release.yml` produces

| Platform | Artifacts |
|---|---|
| Linux | `.deb`, `.AppImage`, `.rpm`, `.sig` files |
| Windows | `.msi`, NSIS `.exe`, `.sig` files |
| macOS | `.dmg`, `.app.tar.gz`, `.sig` files |

All artifacts are attached to the GitHub Release. `latest.json` is uploaded
as a release asset so running instances can find it at the configured
endpoint.

## Updater activation (one-time, maintainer only)

Until a signing keypair is configured, the in-app updater is an
intentional no-op: `tauri.conf.json` has `"pubkey": ""`, which causes the
Tauri updater plugin's `check()` to fail immediately (the guard logs and
skips — no crash, no update prompt).

To activate the updater for a real release:

1. **Generate a keypair** (run once; store the private key offline):

   ```sh
   cargo tauri signer generate
   ```

   This prints a private key (base64) and the corresponding public key.

2. **Store the private key** in the GitHub repository secrets:

   | Secret | Value |
   |---|---|
   | `TAURI_SIGNING_PRIVATE_KEY` | The base64 private key string |
   | `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | The passphrase (or empty string if none) |

   **Never commit the private key to the repository.**

3. **Paste the public key** into `tauri.conf.json`:

   ```json
   "plugins": {
     "updater": {
       "endpoints": [
         "https://github.com/Minutist/minutist/releases/latest/download/latest.json"
       ],
       "pubkey": "<paste-public-key-here>"
     }
   }
   ```

   Commit this change. The updater becomes live on the next release tag.

From that point on `release.yml` signs every bundle with
`TAURI_SIGNING_PRIVATE_KEY`, appends `.sig` files, and `latest.json`
carries the correct signatures. Running instances verify the bundle
against the committed public key before applying.

## Update endpoint

```
https://github.com/Minutist/minutist/releases/latest/download/latest.json
```

The repository is currently private. The no-op pubkey guard covers this:
even if the endpoint 404s or returns an unsigned manifest, the updater
check fails silently (logs at INFO level, never crashes). Activate per
the steps above once the repo is public and a keypair is in place.

## Fallback manifest (off-CI)

`scripts/generate-update-manifest.py` generates a `latest.json` manually
from a local bundle directory. Use this only when testing the updater
flow outside CI.

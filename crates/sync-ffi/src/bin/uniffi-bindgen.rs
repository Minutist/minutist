//! Library-mode UniFFI binding generator for `sync-ffi`.
//!
//! The minutist-mobile build runs this against the cross-compiled `.so` to emit
//! the Kotlin bindings gradle bundles:
//!
//! ```sh
//! cargo run --bin uniffi-bindgen -- generate \
//!     --library target/aarch64-linux-android/release/libsync_ffi.so \
//!     --language kotlin --out-dir <android-src>
//! ```
fn main() {
    uniffi::uniffi_bindgen_main()
}

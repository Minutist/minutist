//! Zero-config artifact acquisition: Gemma-4 vision GGUFs and the PDFium
//! shared library, cached under the OS cache dir.
//!
//! Throwaway spike code: `anyhow`, `eprintln!`, blocking `ureq`. None of this
//! is wired into the app and it deliberately does NOT touch the production
//! model-registry manifest (`models.json`).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

/// Spike cache root: `<os-cache-dir>/minutist-spike/`.
///
/// Falls back to `<target-dir>/.spike-cache/` when the OS cache dir cannot be
/// resolved (rare; e.g. no HOME).
pub fn cache_root() -> Result<PathBuf> {
    let root = match dirs::cache_dir() {
        Some(d) => d.join("minutist-spike"),
        None => {
            let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/.spike-cache");
            eprintln!(
                "WARNING: no OS cache dir; falling back to {}",
                fallback.display()
            );
            fallback
        }
    };
    fs::create_dir_all(&root)
        .with_context(|| format!("creating spike cache dir {}", root.display()))?;
    Ok(root)
}

// ---------------------------------------------------------------------------
// Multi-model vision GGUFs
// ---------------------------------------------------------------------------
//
// Per-model URLs / filenames / size-floors live on each `ModelSpec` in
// `models.rs`; this module just drives the download/cache machinery from them.
// The one model-specific hook that stays here is the app-LM reuse for Gemma-4
// (the app bundles the text-only Gemma LM but never the vision mmproj).

/// Loose lower bound for the app-LM reuse check: a present Gemma-4 LM under the
/// app models dir is only reused if it is at least this big (truncated/partial
/// files are ignored). NOT an exact size — HF may requant.
const LM_MIN_BYTES: u64 = 4_500_000_000; // ~5.34 GB expected

pub struct ModelPaths {
    pub lm: PathBuf,
    pub mmproj: PathBuf,
}

/// Ensure a registered model's LM + mmproj GGUFs exist in the cache, fetching
/// the missing pieces from the model's resolve URLs (size-floor sanity, `.part`
/// + rename, skip when cached). Each model is cached under its own subdir so
/// filenames cannot collide between models.
///
/// As a special case, for the Gemma-4 spec an app-downloaded Gemma-4 LM of the
/// right size is reused if present (the app bundles the text-only LM but never
/// the vision mmproj), so only the mmproj is fetched into the spike cache.
pub fn ensure_model_spec(spec: &crate::models::ModelSpec) -> Result<ModelPaths> {
    let root = cache_root()?;
    // Per-model subdir keyed on the LM cache filename stem; keeps the two
    // models' artifacts from colliding and makes selective deletion easy.
    let model_dir = root.join("vlm").join(model_subdir(spec));
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("creating {}", model_dir.display()))?;

    // LM: reuse an app-downloaded Gemma-4 LM if this is the Gemma spec and one
    // is present and big enough; otherwise fetch from the spec's resolve URL.
    let lm = match (spec.lm_cache_filename.contains("gemma-4"), find_app_gemma4_lm()) {
        (true, Some(existing)) => {
            eprintln!(
                "reusing app-downloaded Gemma-4 LM: {} ({} bytes)",
                existing.display(),
                file_len(&existing).unwrap_or(0)
            );
            existing
        }
        _ => {
            let target = model_dir.join(spec.lm_cache_filename);
            ensure_file(&target, spec.lm_url, spec.lm_min_bytes, "LM GGUF")?;
            target
        }
    };

    // mmproj: ALWAYS fetched into the spike cache (the app never bundles it).
    let mmproj = model_dir.join(spec.mmproj_cache_filename);
    ensure_file(&mmproj, spec.mmproj_url, spec.mmproj_min_bytes, "vision mmproj GGUF")?;

    Ok(ModelPaths { lm, mmproj })
}

/// Stable per-model cache subdir derived from the LM cache filename (extension
/// stripped). e.g. `gemma-4-E4B-it-Q4_K_M` / `PaddleOCR-VL-1.6-q4_k_m`.
fn model_subdir(spec: &crate::models::ModelSpec) -> &'static str {
    spec.lm_cache_filename
        .strip_suffix(".gguf")
        .unwrap_or(spec.lm_cache_filename)
}

/// Look for an already-present Gemma-4 LM under the production app models dir
/// (`<os-data-dir>/minutist/models/llm/**`). Best-effort; returns the first
/// file whose name contains "gemma-4" and "Q4_K_M" and is large enough.
fn find_app_gemma4_lm() -> Option<PathBuf> {
    let data = dirs::data_dir()?;
    let llm_dir = data.join("minutist").join("models").join("llm");
    let mut stack = vec![llm_dir];
    while let Some(dir) = stack.pop() {
        let rd = fs::read_dir(&dir).ok()?;
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
            let looks_right = name.contains("gemma-4")
                && name.ends_with(".gguf")
                && !name.contains("mmproj");
            if looks_right && file_len(&path).unwrap_or(0) >= LM_MIN_BYTES {
                return Some(path);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// PDFium prebuilt (bblanchon/pdfium-binaries GitHub release)
// ---------------------------------------------------------------------------

/// Ensure the platform PDFium shared library is extracted into the cache and
/// return its absolute path (suitable for `Pdfium::bind_to_library`).
pub fn ensure_pdfium() -> Result<PathBuf> {
    let root = cache_root()?;
    let pdfium_dir = root.join("pdfium");
    fs::create_dir_all(&pdfium_dir)
        .with_context(|| format!("creating {}", pdfium_dir.display()))?;

    let (asset, in_archive, lib_name) = pdfium_asset()?;
    let lib_path = pdfium_dir.join(lib_name);
    if lib_path.exists() && file_len(&lib_path).unwrap_or(0) > 1_000_000 {
        eprintln!("PDFium present: {}", lib_path.display());
        return Ok(lib_path);
    }

    let url = format!(
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/{asset}"
    );
    eprintln!("downloading PDFium prebuilt: {url}");
    let tgz = pdfium_dir.join(asset);
    download_to(&url, &tgz, 0, "PDFium archive")?;

    // Extract just the shared library from the gzip tarball.
    extract_member(&tgz, in_archive, &lib_path)
        .with_context(|| format!("extracting {in_archive} from {}", tgz.display()))?;
    let _ = fs::remove_file(&tgz);

    if !lib_path.exists() {
        bail!(
            "PDFium extraction did not produce {} (archive layout changed?)",
            lib_path.display()
        );
    }
    eprintln!("PDFium ready: {}", lib_path.display());
    Ok(lib_path)
}

/// (release asset name, path inside archive, output lib filename) for this host.
fn pdfium_asset() -> Result<(&'static str, &'static str, &'static str)> {
    let arch = if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        bail!("unsupported target_arch for PDFium prebuilt; download manually");
    };

    if cfg!(target_os = "windows") {
        // win arm64 prebuilts exist but x64 is the live-test target.
        Ok((
            if arch == "x64" {
                "pdfium-win-x64.tgz"
            } else {
                "pdfium-win-arm64.tgz"
            },
            "bin/pdfium.dll",
            "pdfium.dll",
        ))
    } else if cfg!(target_os = "macos") {
        Ok((
            if arch == "arm64" {
                "pdfium-mac-arm64.tgz"
            } else {
                "pdfium-mac-x64.tgz"
            },
            "lib/libpdfium.dylib",
            "libpdfium.dylib",
        ))
    } else if cfg!(target_os = "linux") {
        Ok((
            if arch == "arm64" {
                "pdfium-linux-arm64.tgz"
            } else {
                "pdfium-linux-x64.tgz"
            },
            "lib/libpdfium.so",
            "libpdfium.so",
        ))
    } else {
        bail!("unsupported target_os for PDFium prebuilt; download manually")
    }
}

/// Extract a single named member of a .tgz to `out`.
fn extract_member(tgz: &Path, member: &str, out: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let f = fs::File::open(tgz)
        .with_context(|| format!("opening {}", tgz.display()))?;
    let mut archive = Archive::new(GzDecoder::new(f));
    for entry in archive.entries().context("reading tar entries")? {
        let mut entry = entry.context("reading tar entry")?;
        let path = entry.path().context("tar entry path")?;
        // Match the trailing path (archives sometimes prefix with "./").
        let matches = path.to_string_lossy().trim_start_matches("./") == member;
        if matches {
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).context("reading member bytes")?;
            fs::write(out, &buf)
                .with_context(|| format!("writing {}", out.display()))?;
            return Ok(());
        }
    }
    bail!("member {member} not found in {}", tgz.display())
}

// ---------------------------------------------------------------------------
// Download primitives
// ---------------------------------------------------------------------------

/// Ensure `path` exists and is at least `min_bytes`; otherwise (re)download
/// from `url`. A present-but-too-small file is treated as a failed prior
/// download and re-fetched.
fn ensure_file(path: &Path, url: &str, min_bytes: u64, label: &str) -> Result<()> {
    if let Ok(len) = file_len(path) {
        if min_bytes == 0 || len >= min_bytes {
            eprintln!("{label} cached: {} ({len} bytes)", path.display());
            return Ok(());
        }
        eprintln!(
            "{label} at {} is {len} bytes (< {min_bytes} expected); re-downloading",
            path.display()
        );
    }
    eprintln!("downloading {label}: {url}");
    download_to(url, path, min_bytes, label)
}

/// Stream `url` to `path` (to a `.part` file, then rename), logging progress.
/// Verifies the final size against `min_bytes` when non-zero.
fn download_to(url: &str, path: &Path, min_bytes: u64, label: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = path.with_extension("part");

    let resp = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP GET {url} failed (offline or URL moved?): {e}"))?;

    if resp.status() != 200 {
        bail!("HTTP {} for {url}", resp.status());
    }

    let total: Option<u64> = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = resp.into_reader();
    let mut out = fs::File::create(&part)
        .with_context(|| format!("creating {}", part.display()))?;

    let mut buf = vec![0u8; 1 << 20];
    let mut written: u64 = 0;
    let mut last_log = Instant::now();
    let t0 = Instant::now();
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("reading {label} body"))?;
        if n == 0 {
            break;
        }
        use std::io::Write;
        out.write_all(&buf[..n])
            .with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if last_log.elapsed().as_secs() >= 2 {
            log_progress(label, written, total, t0);
            last_log = Instant::now();
        }
    }
    use std::io::Write;
    out.flush().ok();
    drop(out);
    log_progress(label, written, total, t0);

    if min_bytes != 0 && written < min_bytes {
        let _ = fs::remove_file(&part);
        bail!(
            "{label} download truncated: got {written} bytes, expected >= {min_bytes} \
             (network interruption, or the resolve URL returned an HTML error page)"
        );
    }

    fs::rename(&part, path)
        .with_context(|| format!("renaming {} -> {}", part.display(), path.display()))?;
    eprintln!("{label} done: {} ({written} bytes)", path.display());
    Ok(())
}

fn log_progress(label: &str, written: u64, total: Option<u64>, t0: Instant) {
    let mb = written as f64 / 1_048_576.0;
    let secs = t0.elapsed().as_secs_f64().max(0.001);
    let rate = mb / secs;
    match total {
        Some(t) if t > 0 => {
            let pct = written as f64 / t as f64 * 100.0;
            eprintln!(
                "  {label}: {mb:.1} MB / {:.1} MB ({pct:.0}%) @ {rate:.1} MB/s",
                t as f64 / 1_048_576.0
            );
        }
        _ => eprintln!("  {label}: {mb:.1} MB @ {rate:.1} MB/s"),
    }
}

fn file_len(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}

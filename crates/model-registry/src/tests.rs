//! Unit tests for `model-registry`.
//!
//! Test 1: manifest parses cleanly from fixture JSON.
//! Test 2: `list_models` returns `Missing` on empty cache; `Available` after
//!         files with correct SHA are placed.
//! Test 3: `ensure` returns the per-model cache directory when files exist
//!         and hashes match.
//! Test 4: `ensure` returns `AppError::ModelDownload` on SHA mismatch.
//! Test 5: two concurrent `ensure(same_id)` calls download once; both get
//!         the same `Ok(PathBuf)` (wiremock mock endpoint).

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use sha2::{Digest, Sha256};
    use tokio::sync::broadcast;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use meeting_app_common::{
        AppError, ModelFileEntry, ModelId, ModelKind, ModelManifestEntry, ModelStatusState,
    };

    use crate::manifest::parse_manifest;
    use crate::registry::ModelRegistry;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    const FIXTURE_MANIFEST: &[u8] = br#"{
        "version": 1,
        "models": [
            {
                "id": "test-model-q8_0",
                "kind": "asr",
                "display_name": "Test Model Q8_0",
                "license": "apache-2.0",
                "total_size_bytes": 20,
                "files": [
                    {
                        "filename": "model.bin",
                        "url": "http://example.com/model.bin",
                        "size": 20,
                        "sha256": "placeholder"
                    }
                ]
            }
        ]
    }"#;

    fn make_file_content() -> Vec<u8> {
        b"hello fixture world!".to_vec()
    }

    fn sha256_hex(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        format!("{:x}", h.finalize())
    }

    /// Build a single-file `ModelManifestEntry` pointing at `url` with the
    /// SHA of `content`.
    fn make_entry(id: &str, filename: &str, url: &str, content: &[u8]) -> ModelManifestEntry {
        let size = content.len() as u64;
        ModelManifestEntry {
            id: ModelId(id.to_string()),
            kind: ModelKind::Asr,
            display_name: "Test".into(),
            license: "apache-2.0".into(),
            total_size_bytes: size,
            files: vec![ModelFileEntry {
                filename: filename.to_string(),
                url: url.to_string(),
                size,
                sha256: sha256_hex(content),
            }],
        }
    }

    fn registry_with_entry(
        cache_root: PathBuf,
        entry: ModelManifestEntry,
    ) -> ModelRegistry {
        let (tx, _rx) = broadcast::channel(64);
        ModelRegistry::new(cache_root, vec![entry], tx).expect("registry")
    }

    // -----------------------------------------------------------------------
    // Test 1: manifest parses cleanly
    // -----------------------------------------------------------------------

    #[test]
    fn manifest_parses_cleanly() {
        let entries = parse_manifest(FIXTURE_MANIFEST).expect("parse");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id.0, "test-model-q8_0");
        assert_eq!(entries[0].kind, ModelKind::Asr);
        assert_eq!(entries[0].files.len(), 1);
        assert_eq!(entries[0].files[0].filename, "model.bin");
    }

    // -----------------------------------------------------------------------
    // Test 2: list_models — Missing on empty cache; Available after placing files
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn list_models_missing_then_available() {
        let dir = tempfile::tempdir().expect("tempdir");
        let content = make_file_content();
        let entry = make_entry(
            "test-model-q8_0",
            "model.bin",
            "http://example.com/model.bin",
            &content,
        );
        let registry = registry_with_entry(dir.path().to_path_buf(), entry.clone());

        // Empty cache → Missing.
        let statuses = registry.list_models();
        assert_eq!(statuses.len(), 1);
        assert!(
            matches!(statuses[0].status, ModelStatusState::Missing { .. }),
            "expected Missing, got {:?}",
            statuses[0].status
        );

        // Place the file with the correct SHA.
        let model_dir = dir.path().join("asr").join("test-model-q8_0");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.bin"), &content).unwrap();

        // Now list_models should see Available.
        let statuses = registry.list_models();
        assert!(
            matches!(statuses[0].status, ModelStatusState::Available { .. }),
            "expected Available, got {:?}",
            statuses[0].status
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: ensure returns the per-model cache dir when files exist + hash ok
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn ensure_returns_dir_when_files_present_and_hash_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let content = make_file_content();
        let entry = make_entry(
            "test-model-q8_0",
            "model.bin",
            "http://example.com/model.bin",
            &content,
        );
        let model_dir = dir.path().join("asr").join("test-model-q8_0");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.bin"), &content).unwrap();

        let registry = registry_with_entry(dir.path().to_path_buf(), entry);
        let result = registry.ensure(&ModelId("test-model-q8_0".into())).await;
        assert!(result.is_ok(), "ensure failed: {:?}", result.err());
        assert_eq!(result.unwrap(), model_dir);
    }

    // -----------------------------------------------------------------------
    // Test 4: ensure returns AppError::ModelDownload on SHA mismatch
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn ensure_error_on_hash_mismatch_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let correct_content = make_file_content();
        let wrong_content = b"this is not the right file content!";

        // Manifest has the correct SHA but we write the wrong content.
        let entry = make_entry(
            "test-model-q8_0",
            "model.bin",
            "http://example.com/model.bin",
            correct_content.as_slice(),
        );

        let model_dir = dir.path().join("asr").join("test-model-q8_0");
        std::fs::create_dir_all(&model_dir).unwrap();
        // Write wrong content — size matches entry.size so the existing-file
        // fast-path triggers, reads it, and detects the mismatch.
        //
        // We need the on-disk size to equal entry.size (20) so the "file
        // appears complete" branch fires. Make wrong_content the same length.
        let padded: Vec<u8> = wrong_content
            .iter()
            .chain(std::iter::repeat(&b'_'))
            .take(correct_content.len())
            .copied()
            .collect();
        std::fs::write(model_dir.join("model.bin"), &padded).unwrap();

        let registry = registry_with_entry(dir.path().to_path_buf(), entry);
        let result = registry.ensure(&ModelId("test-model-q8_0".into())).await;

        // The registry deletes the bad file and tries to download. Since
        // there's no real server here, it will fail with an Http error —
        // which is still an AppError::ModelDownload. The important thing is
        // that it did NOT return Ok.
        assert!(
            result.is_err(),
            "expected error on SHA mismatch, got Ok"
        );
        match result.unwrap_err() {
            AppError::ModelDownload { .. } => {}
            other => panic!("expected ModelDownload, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: in-flight dedup — two concurrent ensure calls download once
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn in_flight_dedup_downloads_once() {
        let mock_server = MockServer::start().await;
        let content = make_file_content();
        let sha = sha256_hex(&content);

        Mock::given(method("GET"))
            .and(path("/model.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.clone()))
            .expect(1) // exactly one real HTTP request
            .mount(&mock_server)
            .await;

        let dir = tempfile::tempdir().expect("tempdir");
        let url = format!("{}/model.bin", mock_server.uri());
        let entry = ModelManifestEntry {
            id: ModelId("test-model-q8_0".into()),
            kind: ModelKind::Asr,
            display_name: "Test".into(),
            license: "apache-2.0".into(),
            total_size_bytes: content.len() as u64,
            files: vec![ModelFileEntry {
                filename: "model.bin".to_string(),
                url,
                size: content.len() as u64,
                sha256: sha,
            }],
        };

        let (tx, _rx) = broadcast::channel(64);
        let registry = Arc::new(
            ModelRegistry::new(dir.path().to_path_buf(), vec![entry], tx)
                .expect("registry"),
        );

        let id = ModelId("test-model-q8_0".into());
        let reg1 = Arc::clone(&registry);
        let reg2 = Arc::clone(&registry);
        let id1 = id.clone();
        let id2 = id.clone();

        // Launch both tasks concurrently.
        let (r1, r2) = tokio::join!(
            tokio::spawn(async move { reg1.ensure(&id1).await }),
            tokio::spawn(async move { reg2.ensure(&id2).await }),
        );

        let path1 = r1.expect("task1 panic").expect("task1 error");
        let path2 = r2.expect("task2 panic").expect("task2 error");

        // Both should return the same directory.
        assert_eq!(path1, path2);

        // The wiremock expectation of exactly 1 request is verified on drop.
    }
}

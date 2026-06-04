# meeting-app developer task runner.
#
# Thin wrappers over the underlying cargo / npm / script commands so the common
# build / test / clean tasks are one-liners with LIBCLANG_PATH (needed by
# llama-cpp-sys-2's bindgen) already set. Run `make` or `make help` for the list.

# libclang for bindgen. Override via the environment if your LLVM lives elsewhere.
LIBCLANG_PATH ?= /usr/lib/llvm-18/lib
export LIBCLANG_PATH

CARGO  ?= cargo
UI_DIR := ui

.DEFAULT_GOAL := help
.PHONY: help build build-release test test-rust test-ui ui-deps bindings \
        clippy fmt fmt-check render-arch clean clean-all \
        windows-build windows-build-vulkan \
        test-integration test-integration-summary test-integration-asr \
        test-integration-diarize

# Local integration-test config (real model + recording paths). Git-excluded;
# copy tests-local.env.example to create it. Crates whose #[ignore]/env-gated
# tests exercise real models.
TEST_ENV  ?= tests-local.env
INTEG_PKGS := -p summariser -p asr-runtime -p diarizer -p orchestrator -p ipc-bridge

help: ## List the available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Build -----------------------------------------------------------------
build: ## Debug build of the whole workspace
	$(CARGO) build --workspace

build-release: ## Release build of the whole workspace
	$(CARGO) build --workspace --release

# --- Test ------------------------------------------------------------------
test: test-rust test-ui ## Run the full default suite (Rust + UI)

test-rust: ## Rust workspace tests (model/GPU-gated tests skip without env vars)
	$(CARGO) test --workspace

test-ui: ui-deps ## UI type-check + build + Vitest (matches CI)
	cd $(UI_DIR) && npm run build && npm run test

# --- Integration (real models + recordings; needs $(TEST_ENV)) -------------
# Sources $(TEST_ENV) and runs the model-gated #[ignore] tests directly, so
# model-integration bugs (e.g. an unrenderable chat template) surface in WSL
# without a Windows rebuild. `cargo test ARGS=...` narrows to one test.
test-integration: ## Run ALL model-gated integration tests (real models)
	@test -f $(TEST_ENV) || { echo "Missing $(TEST_ENV) — copy tests-local.env.example and adjust"; exit 1; }
	set -a; . ./$(TEST_ENV); set +a; \
		$(CARGO) test $(INTEG_PKGS) $(ARGS) -- --include-ignored

test-integration-summary: ## Gated summariser test (real Gemma LLM)
	@test -f $(TEST_ENV) || { echo "Missing $(TEST_ENV)"; exit 1; }
	set -a; . ./$(TEST_ENV); set +a; \
		$(CARGO) test -p summariser $(ARGS) -- --include-ignored

test-integration-asr: ## Gated ASR test (real Qwen3-ASR model)
	@test -f $(TEST_ENV) || { echo "Missing $(TEST_ENV)"; exit 1; }
	set -a; . ./$(TEST_ENV); set +a; \
		$(CARGO) test -p asr-runtime $(ARGS) -- --include-ignored

test-integration-diarize: ## Gated diarization tests (real ONNX models)
	@test -f $(TEST_ENV) || { echo "Missing $(TEST_ENV)"; exit 1; }
	set -a; . ./$(TEST_ENV); set +a; \
		$(CARGO) test -p diarizer -p orchestrator $(ARGS) -- --include-ignored

ui-deps: ## Install UI dependencies if node_modules is missing
	cd $(UI_DIR) && [ -d node_modules ] || npm ci

# --- Quality ---------------------------------------------------------------
clippy: ## Clippy across the workspace, all targets
	$(CARGO) clippy --workspace --all-targets

fmt: ## Format Rust code
	$(CARGO) fmt --all

fmt-check: ## Check Rust formatting without writing (CI-style)
	$(CARGO) fmt --all --check

# --- Codegen / docs --------------------------------------------------------
bindings: ## Regenerate ui/src/ipc/bindings.ts from the Rust IPC surface
	$(CARGO) run -p meeting-app --bin generate-bindings

render-arch: ## Re-render the C4 SVGs from workspace.dsl (needs Docker)
	scripts/render-architecture.sh

# --- Clean -----------------------------------------------------------------
clean: ## Remove the Rust target/ and the built UI dist
	$(CARGO) clean
	rm -rf $(UI_DIR)/dist scripts/__pycache__

clean-all: clean ## Also remove the UI node_modules
	rm -rf $(UI_DIR)/node_modules

# --- Windows native build (WSL only; needs the Windows MSVC + Rust toolchain) ---
# Drives scripts/build-windows-app.ps1 on the Windows side via powershell.exe.
# The UNC path + C:\Users\anl layout match the other run-*-windows.ps1 scripts;
# adjust WIN_SCRIPT if your checkout/distro differ.
WIN_SCRIPT ?= \\wsl.localhost\Ubuntu\home\anl\meeting-app\scripts\build-windows-app.ps1
WIN_RUN    := powershell.exe -NoProfile -ExecutionPolicy Bypass -File '$(WIN_SCRIPT)'

windows-build: ## Build the portable Windows CPU app (WSL -> Windows MSVC)
	$(WIN_RUN)

windows-build-vulkan: ## Build the portable Windows Vulkan app (WSL -> Windows MSVC)
	$(WIN_RUN) -Features vulkan

# minutist developer task runner.
#
# Thin wrappers over the underlying cargo / npm / script commands so the common
# build / test / clean tasks are one-liners with LIBCLANG_PATH (needed by
# llama-cpp-sys-2's bindgen) already set. Run `make` or `make help` for the list.

# libclang for bindgen. Auto-detects the two common Linux layouts: Debian/
# Ubuntu versioned dirs (/usr/lib/llvm-N/lib) and Arch/Manjaro flat /usr/lib.
# Override via the environment if your LLVM lives elsewhere.
LIBCLANG_PATH ?= $(firstword $(wildcard /usr/lib/llvm-*/lib) $(patsubst %/,%,$(dir $(wildcard /usr/lib/libclang.so))))
export LIBCLANG_PATH

CARGO  ?= cargo
UI_DIR := ui

.DEFAULT_GOAL := help
.PHONY: help build build-release test test-rust test-ui ui-deps bindings \
        clippy fmt fmt-check render-arch clean clean-all \
        windows-build windows-build-vulkan \
        build-free build-free-vulkan \
        test-integration test-integration-summary test-integration-asr \
        test-integration-diarize \
        ci-local

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
	$(CARGO) run -p minutist --bin generate-bindings

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
#
# WIN_SRC_UNC: UNC path to THIS repo as seen from Windows.  Derived automatically
#   via `wslpath -w` (available on any modern WSL2 distro); override if your WSL
#   build has `wslpath` on a non-standard PATH or you want a hard-coded value.
#
# WIN_BUILD_DIR: the Windows-side mirror directory where robocopy stages the
#   source before cargo builds.  Defaults to C:\Users\<your-Windows-user>\meeting-app.
#   Override e.g. WIN_BUILD_DIR=C:\dev\minutist to use a different drive/path.
# WIN_TARGET_DIR: the cargo cache dir (CARGO_TARGET_DIR). Left empty, the PS script
#   derives it from WIN_BUILD_DIR — the canonical live-test dir keeps the shared
#   C:\mt (warm cache + run-path); any other worktree gets its own short cache, so
#   two worktrees building concurrently cannot poison each other's crate rlibs.
#   Override to pin a specific short cache, e.g. WIN_TARGET_DIR=C:\mtc.
WIN_SRC_UNC   ?= $(shell wslpath -w "$(CURDIR)" 2>/dev/null || echo '\\wsl.localhost\Ubuntu\home\anl\meeting-app')
WIN_BUILD_DIR ?= C:\Users\anl\meeting-app
WIN_TARGET_DIR ?=
WIN_SCRIPT    ?= $(WIN_SRC_UNC)\scripts\build-windows-app.ps1
# Captured in WSL (the Windows mirror excludes .git); passed to build.rs so the
# title bar / diagnostic reports identify the exact commit.
GIT_SHA       := $(shell git -C "$(CURDIR)" rev-parse --short HEAD 2>/dev/null || echo unknown)
WIN_RUN       := powershell.exe -NoProfile -ExecutionPolicy Bypass \
                 -File '$(WIN_SCRIPT)' -WslSrc '$(WIN_SRC_UNC)' -BuildDir '$(WIN_BUILD_DIR)' -TargetDir '$(WIN_TARGET_DIR)'

windows-build: ## Build the portable Windows CPU app (WSL -> Windows MSVC)
	$(WIN_RUN) -GitSha '$(GIT_SHA)'

windows-build-vulkan: ## Build the portable Windows Vulkan app (WSL -> Windows MSVC)
	$(WIN_RUN) -Features vulkan -GitSha '$(GIT_SHA)'

# --- Free-tier (no MCP server) local builds --------------------------------
# The free artifact is --no-default-features + an explicit GPU backend.
# VITE_CONNECTED must be unset so the Vite bundler drops the MCP settings
# pane.  Set it explicitly to empty here to prevent shell inheritance of a
# parent VITE_CONNECTED=1.
#
# Both targets build the frontend first (Node is not on Windows; keep the
# same WSL-prebuilt-then-mirror pattern if you adapt these for Windows).

build-free: ## Free-tier debug build (no MCP server, CPU only)
	cd $(UI_DIR) && VITE_CONNECTED= npm run build
	$(CARGO) build --no-default-features

build-free-vulkan: ## Free-tier debug build (no MCP server, Vulkan GPU)
	cd $(UI_DIR) && VITE_CONNECTED= npm run build
	$(CARGO) build --no-default-features --features vulkan

# --- Self-hosted CI runner (docker-managed; see ci/runner/README.md) --------
runner-up: ## Build + start the self-hosted GitHub Actions runner
	docker compose -f ci/runner/docker-compose.yml up -d --build

runner-down: ## Stop the self-hosted runner container
	docker compose -f ci/runner/docker-compose.yml down

runner-logs: ## Tail the self-hosted runner logs
	docker logs -f minutist-github-runner

ci-local: ## Run the Linux CI test suite in the local Docker image (no GitHub round-trip)
	ci/scripts/ci-local.sh

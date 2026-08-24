# Minutist headless sync hub (`minutist-hub`) — container image.
#
# Multi-stage: an Ubuntu 24.04 builder (so the binary links glibc 2.39, the
# current LTS baseline) and a slim Ubuntu runtime holding only the binary and the
# shared libraries it needs. The builder stage also doubles as a way to produce a
# glibc-2.39 binary for bare-metal installs on an LTS host (see packaging/README.md).
#
# Build context is the REPO ROOT (the workspace manifest references every member,
# so the whole source tree must be present to resolve the build). Use the
# Dockerfile-specific ignore file to keep the context small:
#   DOCKER_BUILDKIT=1 docker build -f packaging/Dockerfile -t minutist-hub .

# ---- builder ----------------------------------------------------------------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
# Build deps: clang/libclang for libsql-ffi's bindgen, libopus for audiopus,
# the usual cc toolchain, plus curl/git for rustup.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config cmake clang libclang-dev libopus-dev \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# rustup installs the toolchain pinned by the workspace rust-toolchain.toml
# (iroh 1.0 needs >= 1.91); a minimal profile keeps the layer small.
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /src
COPY . .
# Build only the hub binary and its dependency tree (not the desktop app / ML
# crates, which are workspace members but not dependencies of `headless`).
RUN cargo build --release -p headless && strip target/release/minutist-hub

# ---- runtime ----------------------------------------------------------------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# ca-certificates for the relay's TLS; libopus0 in case audiopus linked it
# dynamically (harmless if the build linked it statically).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libopus0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/minutist-hub /usr/local/bin/minutist-hub

# The data root (device key, account credential, meetings, logs) lives on a mounted volume so
# the device identity survives container recreation.
VOLUME ["/var/lib/minutist-hub"]
# Relay token is supplied at run time: `-e MINUTIST_SYNC_TOKEN=...`.
ENTRYPOINT ["/usr/local/bin/minutist-hub"]
CMD ["--data-dir", "/var/lib/minutist-hub"]

# landa-ai-rag container image (SEP-3). Target: linux/amd64 (pymupdf arm64 wheels are unverified).
#
#   docker build -t landa-ai-rag:local .
#   docker build --build-arg WITH_LIBREOFFICE=true -t landa-ai-rag:local-lo .   # legacy .doc conversion
#   docker build --target test -t landa-ai-rag:test . && docker run --rm landa-ai-rag:test
#       ^ runs scripts/check.sh on the same base image and wheels as the runtime (pip-audit needs network)
#
# Base image: pinned by tag AND digest. The build fails until the placeholder below is replaced:
#   docker buildx imagetools inspect python:3.12-slim-trixie      # copy the index "Digest: sha256:..."
# Prefer the exact patch tag that matches .python-version (e.g. python:3.12.N-slim-trixie@sha256:...)
# and bump tag + digest together on a schedule (security fixes arrive through the base image).
ARG PYTHON_IMAGE=python:3.12-slim-trixie@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

# ---------------------------------------------------------------------------------------------
# build: virtualenv with the hash-locked runtime dependencies only.
# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INPUT=1

RUN python -m venv /opt/venv

COPY requirements.lock /build/requirements.lock

# --require-hashes: every artifact must match requirements.lock.
# --no-deps:        the lock is a complete pip-compile output (compiled on Windows); re-resolving on
#                   Linux could request marker-only dependencies that are not hash-pinned.
# --only-binary:    no compiler in this image; fail fast if a pinned version has no manylinux wheel.
# pip check:        fail if the Windows-compiled lock misses a Linux-only dependency.
RUN /opt/venv/bin/python -m pip install --require-hashes --no-deps --only-binary=:all: \
        -r /build/requirements.lock \
 && /opt/venv/bin/python -m pip check

# ---------------------------------------------------------------------------------------------
# venv: runtime copy of the virtualenv without pip (nothing installs at runtime).
# ---------------------------------------------------------------------------------------------
FROM build AS venv
RUN /opt/venv/bin/python -m pip uninstall --yes pip

# ---------------------------------------------------------------------------------------------
# test (optional target, never the default): quality gate inside a Linux container.
# ---------------------------------------------------------------------------------------------
FROM build AS test
COPY requirements-dev.lock /build/requirements-dev.lock
# requirements-dev.lock leaves pip/setuptools unpinned (pip-tools depends on them): --no-deps.
RUN /opt/venv/bin/python -m pip install --require-hashes --no-deps --only-binary=:all: \
        -r /build/requirements-dev.lock
WORKDIR /src
COPY . /src
# scripts/check.sh expects <repo>/.venv-dev/bin/python.
RUN ln -s /opt/venv /src/.venv-dev \
 && chown -R 10001:10001 /src
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp
USER 10001:10001
CMD ["sh", "scripts/check.sh"]

# ---------------------------------------------------------------------------------------------
# runtime (default target): non-root, read-only-rootfs friendly, no build tools, no pip.
# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG WITH_LIBREOFFICE=false
ARG APP_UID=10001
ARG GIT_SHA=unknown

# LibreOffice is only used to convert legacy .doc uploads (app/services/ingestion/extract.py extract_doc); without it the
# service falls back to byte decoding. The package version follows the pinned base image's Debian
# release; pin it via snapshot.debian.org if bit-for-bit reproducibility is required.
RUN set -eux; \
    case "${WITH_LIBREOFFICE}" in \
        true) \
            apt-get update; \
            apt-get install -y --no-install-recommends libreoffice-writer-nogui; \
            apt-get clean; \
            rm -rf /var/lib/apt/lists/*; \
            ;; \
        false) ;; \
        *) echo "WITH_LIBREOFFICE must be true or false" >&2; exit 1 ;; \
    esac; \
    groupadd --system --gid "${APP_UID}" app; \
    useradd --system --uid "${APP_UID}" --gid app --home-dir /nonexistent --no-create-home \
        --shell /usr/sbin/nologin app; \
    rm -rf /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
           /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12

COPY --from=venv /opt/venv /opt/venv

WORKDIR /app
COPY app/ /app/app/

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TMPDIR=/tmp \
    HOME=/tmp \
    AI_RAG_ENV=production \
    AI_RAG_HOST=0.0.0.0 \
    AI_RAG_PORT=8010 \
    AI_RAG_WORKERS=1

# Bytecode is compiled at build time because the root filesystem is read-only at runtime
# (the service code is ~45k lines). Code stays root-owned and read-only for the app user.
RUN python -m compileall -q -j 0 --invalidation-mode unchecked-hash /app/app

LABEL org.opencontainers.image.title="landa-ai-rag" \
      org.opencontainers.image.source="https://github.com/letrungtin123/landa-ai-rag-be-custom" \
      org.opencontainers.image.revision="${GIT_SHA}"
ENV AI_RAG_BUILD_SHA=${GIT_SHA}

USER ${APP_UID}:${APP_UID}
EXPOSE 8010

# Liveness only (/healthz never touches dependencies). Python stdlib, no curl; proxies bypassed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-I", "-c", "import os, sys, urllib.request as u; port = os.environ.get('AI_RAG_PORT', '8010'); opener = u.build_opener(u.ProxyHandler({})); sys.exit(0 if opener.open('http://127.0.0.1:' + port + '/healthz', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM
CMD ["python", "-m", "app"]

#!/usr/bin/env bash
# Health-gated update of the ai-rag container with automatic rollback (server B, one instance).
#
#   sudo ./deploy.sh sha-<40-hex-git-sha>      # immutable tag pushed by CI
#   sudo ./deploy.sh @sha256:<digest>          # or an image digest from the CI job summary
#
# Steps: ECR login -> pull -> recreate ai-rag only -> wait /readyz -> smoke through nginx TLS
#        (healthz, readyz, 401 unsigned, 200 HMAC-signed) -> commit release.env.
# Any failure: recreate the previous image, wait /readyz, exit 1. nginx is not touched.
# Downtime: the single instance restarts; in-flight requests get AI_RAG_SHUTDOWN_GRACE_SECONDS to
# drain and the backend retries SERVICE_BUSY/connection errors. Deploy outside long IDM runs.
# Run fetch-secrets.sh first when secrets changed (the recreate picks up the new env file).
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
release_env="${RELEASE_ENV:-/etc/landa-ai/release.env}"
ready_timeout="${READY_TIMEOUT_SECONDS:-240}"
history_log="${DEPLOY_HISTORY_LOG:-/var/log/landa-ai-deploy.log}"

die() { echo "deploy: $*" >&2; exit 1; }
[ $# -eq 1 ] || die "usage: $0 <sha-<git-sha>|v<semver>|@sha256:<digest>>"
[ -r "$release_env" ] || die "missing $release_env (copy release.env.example)"

# Read values without sourcing/exporting: exported shell variables would override --env-file.
release_var() { sed -n "s/^$1=//p" "$release_env" | tail -n 1; }
repo=$(release_var AI_RAG_IMAGE_REPO)
previous_image=$(release_var AI_RAG_IMAGE)
region=$(release_var AWS_REGION)
tls_name=$(release_var AI_RAG_TLS_SERVER_NAME)
[ -n "$repo" ] && [ -n "$previous_image" ] && [ -n "$region" ] && [ -n "$tls_name" ] \
    || die "release.env needs AI_RAG_IMAGE_REPO, AI_RAG_IMAGE, AWS_REGION, AI_RAG_TLS_SERVER_NAME"

ref="$1"
case "$ref" in
    @sha256:*)
        [[ "$ref" =~ ^@sha256:[0-9a-f]{64}$ ]] || die "invalid digest"
        new_image="${repo}${ref}" ;;
    *)
        [[ "$ref" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid tag"
        new_image="${repo}:${ref}" ;;
esac
[ "$new_image" != "$previous_image" ] || die "$new_image is already deployed"

compose() {
    local env_file="$1"; shift
    env -u AI_RAG_IMAGE -u AI_RAG_PREVIOUS_IMAGE \
        docker compose --project-directory "$here" -f "$here/docker-compose.yml" --env-file "$env_file" "$@"
}

wait_ready() {
    local deadline=$((SECONDS + ready_timeout))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -fsS -o /dev/null --max-time 3 --noproxy '*' http://127.0.0.1:8010/readyz; then
            return 0
        fi
        sleep 3
    done
    return 1
}

smoke() {
    compose "$1" exec -T ai-rag python - "https://${tls_name}:8443" \
        --cafile /etc/landa-ai/tls/internal-ca.pem < "$here/smoke.py"
}

next_env=$(mktemp "${release_env}.next.XXXXXX")
trap 'rm -f -- "$next_env"' EXIT
awk -v img="$new_image" -v prev="$previous_image" '
    /^AI_RAG_IMAGE=/          { print "AI_RAG_IMAGE=" img; next }
    /^AI_RAG_PREVIOUS_IMAGE=/ { print "AI_RAG_PREVIOUS_IMAGE=" prev; seen = 1; next }
    { print }
    END { if (!seen) print "AI_RAG_PREVIOUS_IMAGE=" prev }
' "$release_env" > "$next_env"
chmod --reference="$release_env" "$next_env"

rollback() {
    echo "deploy: FAILED ($1); rolling back to $previous_image" >&2
    compose "$release_env" up -d --no-deps ai-rag || true
    if wait_ready; then
        echo "deploy: rollback is ready" >&2
    else
        echo "deploy: ROLLBACK NOT READY - escalate (docker compose logs ai-rag)" >&2
    fi
    printf '%s rollback from=%s to=%s reason=%s\n' "$(date -u +%FT%TZ)" "$new_image" "$previous_image" "$1" \
        >> "$history_log" 2>/dev/null || true
    exit 1
}

echo "deploy: $previous_image -> $new_image"
aws ecr get-login-password --region "$region" \
    | docker login --username AWS --password-stdin "${repo%%/*}" >/dev/null
docker pull --quiet "$new_image" >/dev/null || die "pull failed for $new_image (nothing changed)"

compose "$next_env" up -d --no-deps ai-rag || rollback "compose up"
wait_ready || rollback "readyz not 200 within ${ready_timeout}s"
smoke "$next_env" || rollback "smoke test"

mv -f -- "$next_env" "$release_env"
trap - EXIT
printf '%s deploy from=%s to=%s\n' "$(date -u +%FT%TZ)" "$previous_image" "$new_image" \
    >> "$history_log" 2>/dev/null || true
echo "deploy: $new_image is live"

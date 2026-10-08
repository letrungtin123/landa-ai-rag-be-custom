#!/usr/bin/env bash
# Render the ai-rag container env file from AWS Secrets Manager (+ optional SSM Parameter Store).
#
#   sudo AWS_REGION=<aws-region> AI_RAG_SECRET_ID=<secret-name-or-arn> \
#        [AI_RAG_SSM_PATH=/landa/ai-rag/prod/] [AI_RAG_ENV_FILE=/etc/landa-ai/ai-rag.env] ./fetch-secrets.sh
#
# * Secret: a JSON object {"DATABASE_URL": "...", "AI_RAG_SERVICE_HMAC_SECRETS": "...", ...}.
# * SSM (optional): parameters under AI_RAG_SSM_PATH; the last path segment is the variable name
#   (/landa/ai-rag/prod/AI_RAG_AUTH_MODE -> AI_RAG_AUTH_MODE). SecureString is decrypted.
# * Output: KEY='value' lines (single quotes stop docker compose from interpolating "$"), mode 0600,
#   replaced atomically. Values are never printed, only variable names.
# * Credentials: the instance role (IMDSv2). Needs secretsmanager:GetSecretValue (+ kms:Decrypt for a
#   customer-managed key) and, with SSM, ssm:GetParametersByPath on the path.
# Containers only see a new file after they are recreated (deploy.sh, or `docker compose up -d`).
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${AI_RAG_SECRET_ID:?set AI_RAG_SECRET_ID (Secrets Manager name or ARN)}"
ssm_path="${AI_RAG_SSM_PATH:-}"
out="${AI_RAG_ENV_FILE:-/etc/landa-ai/ai-rag.env}"

command -v aws >/dev/null || { echo "aws CLI v2 is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required (preinstalled on Amazon Linux 2023)" >&2; exit 1; }

umask 077
out_dir=$(dirname -- "$out")
install -d -m 0750 -- "$out_dir"
work=$(mktemp -d "$out_dir/.fetch-secrets.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

aws secretsmanager get-secret-value \
    --region "$AWS_REGION" --secret-id "$AI_RAG_SECRET_ID" \
    --query SecretString --output text > "$work/secret.json"

if [ -n "$ssm_path" ]; then
    aws ssm get-parameters-by-path \
        --region "$AWS_REGION" --path "$ssm_path" --recursive --with-decryption \
        --output json > "$work/params.json"
else
    printf '{"Parameters": []}' > "$work/params.json"
fi

python3 - "$work/secret.json" "$work/params.json" "$work/ai-rag.env" <<'PY'
import json
import re
import sys

secret_path, params_path, out_path = sys.argv[1:4]
NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
KID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Pinned by docker-compose.yml `environment:`; a value here would be silently overridden.
PINNED = {
    "AI_RAG_ENV", "AI_RAG_HOST", "AI_RAG_PORT", "AI_RAG_WORKERS", "AI_RAG_PROXY_HEADERS",
    "AI_RAG_FORWARDED_ALLOW_IPS", "AI_RAG_KEEP_ALIVE_TIMEOUT_SECONDS", "TMPDIR",
}
# SEP-1: storage access is a backend-signed URL; the service key is not needed on server B.
REQUIRED = ["DATABASE_URL", "AI_RAG_AUTH_MODE", "AI_RAG_SERVICE_HMAC_SECRETS"]
errors = []


def fail(message):
    errors.append(message)


with open(secret_path, encoding="utf-8") as handle:
    try:
        secret = json.load(handle)
    except json.JSONDecodeError:
        secret = None
if not isinstance(secret, dict):
    sys.exit("secret must be a JSON object of NAME -> string (SecretString)")

values = {}
for key, value in secret.items():
    if not isinstance(value, str):
        fail(f"{key}: secret value must be a string")
        continue
    values[key] = value

with open(params_path, encoding="utf-8") as handle:
    for parameter in json.load(handle).get("Parameters", []):
        key = parameter["Name"].rstrip("/").rsplit("/", 1)[-1]
        if key in values:
            fail(f"{key}: defined in both the secret and SSM")
            continue
        values[key] = parameter["Value"]

for key, value in values.items():
    if not NAME.match(key):
        fail(f"invalid variable name {key!r}")
    if any(ch in value for ch in "\n\r\0'"):
        fail(f"{key}: value contains a newline, NUL or single quote (URL-encode it)")
for key in sorted(set(values) & PINNED):
    print(f"warning: {key} is pinned by docker-compose.yml; the value here is ignored", file=sys.stderr)
for key in REQUIRED:
    if not values.get(key, "").strip():
        fail(f"{key}: required")
if not (values.get("AI_RAG_STORAGE_ALLOWED_ORIGINS", "").strip() or values.get("SUPABASE_URL", "").strip()):
    fail("AI_RAG_STORAGE_ALLOWED_ORIGINS: required (origin of the backend-signed storage URLs)")
if values.get("SUPABASE_SERVICE_KEY"):
    print("warning: SUPABASE_SERVICE_KEY is set; since SEP-1 server B does not need it, remove it", file=sys.stderr)

auth_mode = values.get("AI_RAG_AUTH_MODE", "")
if auth_mode and auth_mode != "hmac":
    fail("AI_RAG_AUTH_MODE must be hmac on server B")
if values.get("AI_RAG_SERVICE_TOKEN"):
    print("warning: AI_RAG_SERVICE_TOKEN is set but unused with AI_RAG_AUTH_MODE=hmac; remove it", file=sys.stderr)

hmac_value = values.get("AI_RAG_SERVICE_HMAC_SECRETS", "")
if hmac_value.strip():
    seen = set()
    for index, entry in enumerate(hmac_value.split(","), start=1):
        kid, sep, key_secret = entry.strip().partition(":")
        if not sep or not KID.match(kid) or len(key_secret) < 16 or kid in seen:
            fail(f"AI_RAG_SERVICE_HMAC_SECRETS entry #{index}: expected unique key_id:secret, secret >= 16 chars")
        elif len(key_secret) < 32:
            print(f"warning: AI_RAG_SERVICE_HMAC_SECRETS entry #{index} ({kid}) is shorter than 32 chars", file=sys.stderr)
        seen.add(kid)

database_url = values.get("DATABASE_URL", "")
if database_url and ("sslmode=verify-full" not in database_url or "sslrootcert=" not in database_url):
    fail("DATABASE_URL must contain sslmode=verify-full and sslrootcert=/etc/landa-ai/tls/pg-ca.pem")

if errors:
    for message in errors:
        print(f"error: {message}", file=sys.stderr)
    sys.exit(1)

with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("# Rendered by fetch-secrets.sh. Do not edit; do not copy.\n")
    for key in sorted(values):
        handle.write(f"{key}='{values[key]}'\n")
print(f"rendered {len(values)} variables: {', '.join(sorted(values))}")
PY

chmod 0600 "$work/ai-rag.env"
chown root:root "$work/ai-rag.env" 2>/dev/null || true
mv -f -- "$work/ai-rag.env" "$out"
echo "wrote $out (0600). Recreate the ai-rag container to apply it."

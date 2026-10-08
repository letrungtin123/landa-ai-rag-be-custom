# Server B (AI) runbook — landa-ai-rag on AWS EC2

Scope: SEP-3 / INF-1 of `AI-ID PLAN/AI_ID_PROD_HARDENING_AND_SEPARATION_PLAN_2026-10-08.md`.
One instance of the Python AI service in Docker, behind nginx TLS on 8443, in a private subnet of
the same VPC as server A (Node backend + Supabase/Postgres). Out to the internet only for Gemini
(via NAT). Placeholders look like `<...>`; nothing in this directory contains a secret.

```
server A (Node BE)  --HTTPS 8443 + HMAC-->  nginx (server B)  --HTTP-->  ai-rag 172.30.10.10:8010
                                                                            |  5432 TLS verify-full -> Postgres (A)
                                                                            |  Kong/Storage (A, until SEP-1)
                                                                            '- 443 via NAT -> Gemini
```

| File | Purpose |
|---|---|
| `../../Dockerfile` | Multi-stage image: hash-locked venv, non-root uid 10001, read-only-rootfs ready, HEALTHCHECK `/healthz` |
| `../../.github/workflows/ci.yml` | check.sh → build → container smoke → SBOM → Trivy → ECR push (main, `v*`) via OIDC |
| `docker-compose.yml` | `ai-rag` + `nginx` services, hardening, awslogs, fixed bridge network |
| `nginx/ai-rag.conf` | TLS 8443, 960 s timeouts, 25 MiB bodies, no path rewrite, upstream keep-alive 60 s, JSON access log |
| `env.example` | Every container variable name, source and meaning |
| `release.env.example` | Non-secret compose variables (image refs, region, log groups, TLS file paths) |
| `fetch-secrets.sh` | Secrets Manager (+ SSM) → `/etc/landa-ai/ai-rag.env` (0600), validated, values never printed |
| `deploy.sh` | Health-gated update with automatic rollback |
| `smoke.py` | Runs inside the container: healthz, readyz, 401 unsigned, 200 HMAC-signed (through nginx TLS) |

Shorthand used below (run as root on server B):

```sh
dc() { env -u AI_RAG_IMAGE -u AI_RAG_PREVIOUS_IMAGE docker compose \
  --project-directory /opt/landa-ai/deploy/server-b --env-file /etc/landa-ai/release.env "$@"; }
```

Never `export` compose variables (AI_RAG_IMAGE, ...) in the shell: shell values override `--env-file`.

## 1. Provisioning checklist (server B)

**Instance**
- [ ] Amazon Linux 2023, **x86_64** (pymupdf arm64 wheels are unverified). Start at 2 vCPU / 8 GiB
      (m6i.large or t3.large); the container is capped at `AI_RAG_MEM_LIMIT=3g` incl. a 512 MiB `/tmp` tmpfs.
- [ ] Private subnet in the **same VPC/region** as server A (D5), no public IP, EBS gp3 ≥ 30 GiB encrypted.
- [ ] IMDSv2 required, **hop limit 1** (containers cannot reach instance credentials; dockerd/awslogs can).
- [ ] Access through SSM Session Manager only; no SSH key, no port 22.
- [ ] Packages: `dnf install -y docker` and the Docker Compose v2 plugin (needs a download: approve and
      verify its checksum first), `systemctl enable --now docker`. aws CLI v2, python3, curl, chrony
      ship with AL2023.
- [ ] `/etc/docker/daemon.json`: `{"live-restore": true, "no-new-privileges": true}` then restart docker.

**Clock (HMAC)** — requests are rejected when `|now − X-Landa-Timestamp| > AI_RAG_AUTH_CLOCK_SKEW_SECONDS`
(300 s) and request IDs are remembered for `AI_RAG_AUTH_REPLAY_TTL_SECONDS` (600 s).
- [ ] chrony enabled on **both** servers (AL2023 default source: Amazon Time Sync 169.254.169.123,
      link-local, not subject to security groups). Check `chronyc tracking` (offset ≪ 1 s) and
      `chronyc sources` (one `^*` source). Containers use the host clock.

**IAM instance role** (least privilege, resources by ARN)
- [ ] `ecr:GetAuthorizationToken` (`*`); `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`,
      `ecr:BatchCheckLayerAvailability` on the `landa-ai-rag` repository.
- [ ] `secretsmanager:GetSecretValue` on the AI secret; `kms:Decrypt` on its key if customer-managed.
- [ ] `ssm:GetParametersByPath` on `arn:aws:ssm:<aws-region>:<account-id>:parameter/landa/ai-rag/prod*` (if used).
- [ ] `logs:CreateLogStream`, `logs:PutLogEvents` on both log groups.
- [ ] `AmazonSSMManagedInstanceCore` (Session Manager); CloudWatch agent policy if host metrics are wanted.

**Security groups**
- [ ] `sg-ai-b` inbound: TCP **8443 from `sg-be-a` only**. Nothing else (8010 is bound to 127.0.0.1).
- [ ] `sg-ai-b` outbound: TCP **5432 → `sg-be-a`**; TCP **`<kong-port>` → `sg-be-a`** (Storage, until
      SEP-1 signed URLs); TCP **443 → 0.0.0.0/0** (through the NAT gateway: Gemini
      `generativelanguage.googleapis.com`, plus ECR/S3/CloudWatch/Secrets Manager/SSM/STS unless VPC
      endpoints exist). Prefer interface/gateway VPC endpoints for the AWS services so that only Gemini
      leaves through NAT; optionally restrict NAT egress by domain with AWS Network Firewall.
- [ ] `sg-be-a` (server A): inbound 5432 and `<kong-port>` from `sg-ai-b`; outbound 8443 → `sg-ai-b`.

**Host layout**
- [ ] `/opt/landa-ai/deploy/server-b/` — this directory from the release commit (`git archive`), root-owned,
      `deploy.sh` and `fetch-secrets.sh` mode 0755.
- [ ] `/etc/landa-ai/` (0750 root): `release.env` (0640), `ai-rag.env` (0600, rendered), `tls/`.
- [ ] Log groups (awslogs does not create them): `aws logs create-log-group --log-group-name /landa/ai-rag/prod/app`
      and `/landa/ai-rag/prod/nginx`, then `aws logs put-retention-policy --retention-in-days 30` on both.

## 2. Internal TLS (server A → B)

- Issue a server certificate from the internal CA (AWS Private CA or an offline OpenSSL CA). SAN =
  the name server A uses, ideally a Route 53 private-zone name (e.g. `ai-rag.<private-zone>`) rather
  than the EC2 `ip-…compute.internal` name. Put the same name in `AI_RAG_TLS_SERVER_NAME`.
- Files on B: `/etc/landa-ai/tls/server.crt` (chain, 0644), `server.key` (`chown root:101`, `chmod 0640`;
  nginx-unprivileged runs as uid 101), `internal-ca.pem` (0644, used by smoke.py), and `pg-ca.pem`
  (0644, the CA of server A's Postgres certificate).
- Server A: the same `internal-ca.pem` in `NODE_EXTRA_CA_CERTS` (read only at Node process start).
- Renewal: replace the files, then `dc exec nginx nginx -s reload` (graceful; long requests continue).
- Optional mTLS: uncomment `ssl_client_certificate`/`ssl_verify_client` in `nginx/ai-rag.conf` and
  give the backend a client certificate.

## 3. Secrets and configuration

- Secrets Manager secret `<ai-rag-secret-name>` (JSON, KMS-encrypted): `DATABASE_URL`,
  `AI_RAG_SERVICE_HMAC_SECRETS`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (the last two only until SEP-1).
- SSM path `/landa/ai-rag/prod/` (optional): non-secret tunables, e.g. `AI_RAG_AUTH_MODE=hmac`,
  limiter/deadline overrides. A name in both sources is an error.
- Render: `sudo AWS_REGION=<aws-region> AI_RAG_SECRET_ID=<ai-rag-secret-name> AI_RAG_SSM_PATH=/landa/ai-rag/prod/ /opt/landa-ai/deploy/server-b/fetch-secrets.sh`.
  It fails on missing required values, `AI_RAG_AUTH_MODE` other than `hmac`, malformed HMAC pairs, or a
  `DATABASE_URL` without `sslmode=verify-full` + `sslrootcert`. Containers see changes only after a recreate.
- `docker inspect` shows container env to anyone with Docker access: only root may use Docker on B.

**HMAC key rotation (zero downtime)**
1. Secret: `AI_RAG_SERVICE_HMAC_SECRETS=<old-kid>:<old>,<new-kid>:<new>` → fetch-secrets →
   `dc up -d --force-recreate --no-deps ai-rag` → wait for `curl -fsS http://127.0.0.1:8010/readyz`.
2. Server A: `AI_RAG_SERVICE_HMAC_KEY_ID=<new-kid>`, `AI_RAG_SERVICE_HMAC_SECRET=<new>` → restart the
   3 backend processes.
3. Remove `<old-kid>` from the secret → fetch-secrets → recreate as in step 1.

## 4. First deploy

1. Complete §1–§3. CI has pushed `…/landa-ai-rag:sha-<git-sha>` (job summary shows tag and digest).
2. `cp release.env.example /etc/landa-ai/release.env`; fill every `<…>`, set `AI_RAG_IMAGE` to the
   CI tag, pin `NGINX_IMAGE` by digest.
3. Render secrets (§3).
4. `aws ecr get-login-password --region <aws-region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<aws-region>.amazonaws.com`
5. `dc config --quiet && dc pull && dc up -d`
6. `until curl -fsS http://127.0.0.1:8010/readyz; do sleep 3; done` (a failing DB connection at
   startup crashes the process today and Docker restarts it; see §9).
7. `dc exec -T ai-rag python - https://<server-b-name>:8443 --cafile /etc/landa-ai/tls/internal-ca.pem < smoke.py`
   → four `PASS` lines.
8. From server A: `curl --cacert <internal-ca.pem> https://<server-b-name>:8443/healthz` → `{"status":"ok"}`;
   then point the staging backend at B (§7) and run index + chat + one IDM run (INF-1 exit).
9. Enable the readyz probe and alarms (§6).

## 5. Update (health-gated) and rollback

```sh
cd /opt/landa-ai/deploy/server-b
sudo ./deploy.sh sha-<git-sha>          # or: sudo ./deploy.sh @sha256:<digest>
```

`deploy.sh` logs in to ECR, pulls, recreates **only** `ai-rag` with the new image, waits up to 240 s
for `/readyz` = 200, runs `smoke.py` through nginx TLS, and only then rewrites `release.env`
(`AI_RAG_IMAGE`, `AI_RAG_PREVIOUS_IMAGE`). On any failure it recreates the previous image, waits for
`/readyz` and exits 1. History: `/var/log/landa-ai-deploy.log`.

- The single instance restarts: in-flight requests get `AI_RAG_SHUTDOWN_GRACE_SECONDS` (60 s) before
  uvicorn stops, `stop_grace_period` (75 s) before SIGKILL. Deploy outside long IDM/index runs.
- Deploy order across servers: **AI first** (backward-compatible contract), then the backend.
- Manual rollback at any time: `sudo ./deploy.sh <previous tag or @digest from release.env>`.
- nginx config change: update the file, `dc exec nginx nginx -t && dc exec nginx nginx -s reload`.
- Image cleanup: prune old images by hand, never the one in `AI_RAG_PREVIOUS_IMAGE`
  (`docker image prune -a` would delete it because no container uses it).
- LibreOffice variant (legacy `.doc`): build with `--build-arg WITH_LIBREOFFICE=true` (CI variable
  `WITH_LIBREOFFICE=true`). `/tmp` is mounted `noexec`; if conversion fails on that, drop `noexec`
  from the ai-rag tmpfs.

## 6. Logs, metrics, alarms

Both containers log JSON lines through the awslogs driver (app: `/landa/ai-rag/prod/app`, written to
stderr by `app.core.logging`; nginx: `/landa/ai-rag/prod/nginx`, path only, no query strings, bodies or
`X-Landa-*` headers). `docker compose logs` keeps working (Docker dual logging).

`/metrics` requires HMAC auth, so a stock Prometheus/CloudWatch-agent scraper cannot read it; alarms use
CloudWatch Logs metric filters (exact counters such as `ai_rag_provider_calls_total` stay available
on demand via a signed request, e.g. `smoke.py`).

| Alarm | Log group | Metric filter pattern | Suggested threshold |
|---|---|---|---|
| App 5xx | app | `{ $.event = "http_request_completed" && $.status >= 500 }` (+ `{ $.event = "http_request_completed" }` as denominator) | > 5 % or ≥ 5 in 5 min |
| Proxy 5xx (502/504: app down or > 960 s) | nginx | `{ $.status >= 500 }` | ≥ 3 in 5 min |
| readyz failing | nginx (probe below) | `{ $.path = "/readyz" && $.status != 200 }` | ≥ 3 of 5 min |
| readyz probe missing | nginx | `{ $.path = "/readyz" }` | < 3 in 5 min, missing data = breaching |
| DB unreachable | app | `{ $.event = "readiness_database_unavailable" }` | ≥ 1 in 5 min |
| Provider outcome ≠ success | app | `{ $.event = "provider_request_failed" \|\| $.event = "provider_timeout" \|\| $.event = "provider_unavailable" \|\| $.event = "provider_quota_exhausted" \|\| $.event = "ai_provider_*" }` | ≥ 10 in 10 min; quota ≥ 1 |
| Container restarts | app | `{ $.event = "service_started" }` | > 1 in 15 min outside a deploy |
| Deadlines / unhandled | app | `{ $.event = "request_deadline_exceeded" }`, `{ $.event = "unhandled_request_error" }` | ≥ 1 |
| Host | EC2 / CW agent | `StatusCheckFailed`, `mem_used_percent`, `disk_used_percent` | standard |

Re-check provider event names after STB-1 (log formatter fix) against real log lines.

readyz probe (one nginx log line per minute, through TLS, which also feeds the two readyz alarms):

```ini
# /etc/systemd/system/ai-rag-readyz-probe.service
[Service]
Type=oneshot
ExecStart=/usr/bin/curl -fsS -o /dev/null --max-time 5 --cacert /etc/landa-ai/tls/internal-ca.pem \
  --resolve <server-b-name>:8443:<server-b-private-ip> https://<server-b-name>:8443/readyz

# /etc/systemd/system/ai-rag-readyz-probe.timer
[Timer]
OnCalendar=*-*-* *:*:00
[Install]
WantedBy=timers.target
```

`systemctl daemon-reload && systemctl enable --now ai-rag-readyz-probe.timer`

## 7. Server A requirements

- **Postgres TLS**: `ssl = on` with a certificate whose SAN is the host in B's `DATABASE_URL`; give B
  the issuing CA as `pg-ca.pem`. `listen_addresses` includes the private IP. Connect B to Postgres
  directly or to Supavisor in **session** mode; transaction mode (6543) needs statement cache 0 (SEP-1).
- **pg_hba.conf**: `hostssl postgres landa_ai_rag <server-b-private-ip>/32 scram-sha-256` and no
  non-SSL `host` line that matches B; reload Postgres. Role `landa_ai_rag` from SEP-2 (approved SQL).
- **Storage/Kong**: reachable only on the private IP from `sg-ai-b` (until SEP-1 replaces the service
  key with signed URLs). If Kong serves HTTPS with the internal CA, confirm the Python storage client
  trusts that CA; otherwise keep it private-network-only HTTP until SEP-1.
- **Backend env** (process env, e.g. the PM2 ecosystem file):
  - `AI_RAG_SERVICE_URL=https://<server-b-name>:8443` — origin only, no path: the backend signs
    `url.pathname`, nginx forwards it unchanged.
  - `NODE_EXTRA_CA_CERTS=<path>/internal-ca.pem`
  - `AI_RAG_SERVICE_HMAC_KEY_ID=<kid>`, `AI_RAG_SERVICE_HMAC_SECRET=<secret>` (one pair present in B's
    `AI_RAG_SERVICE_HMAC_SECRETS`); `AI_RAG_SERVICE_TOKEN` unset.
  - `AI_RAG_REQUEST_TIMEOUT_MS` ≤ 900000 (below nginx's 960 s).
  - `BIND_HOST=127.0.0.1` behind A's own nginx (D7) and `TRUSTED_PROXY_CIDRS=127.0.0.1/32`.
- chrony on A as well (§1).

## 8. Quality gate in a Linux container

`docker build --target test -t landa-ai-rag:test . && docker run --rm landa-ai-rag:test` runs
`scripts/check.sh` on the runtime base image and wheels (pip-audit needs network). The Node
cross-language test needs `../landa-backend` and fails there; CI deselects it unless the backend repo
is checked out (see `ci.yml`).

## 9. Known gaps (track before INF-2)

- Startup connects the DB pool before serving (`app/main.py` `startup()`, fixed pool 1..8): if Postgres is
  unreachable at boot the process exits and Docker restarts it; `/healthz` is not served meanwhile (SEP-1 #1).
- `SUPABASE_URL`/`SUPABASE_SERVICE_KEY` still required (SEP-1 #3). Outbound allowlist in code: SEP-1 #4.
- `uvloop` is not in `requirements.lock` (compiled on Windows; the `uvicorn[standard]` marker excludes it).
- `AI_RAG_EXTRACTION_EXECUTOR=process` would use `fork` on Linux (`app/core/concurrency.py`); keep `thread`.
- HMAC replay cache is in-process: one instance, `AI_RAG_WORKERS=1`. Scaling needs a shared store + ALB
  (idle timeout ≥ 960 s).
- Pin GitHub Actions and the Postgres CI service by SHA/digest; pin `PYTHON_IMAGE` and `NGINX_IMAGE`.

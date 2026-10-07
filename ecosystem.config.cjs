// Entrypoint: `python -m app` reads host/port/workers/graceful-shutdown from
// validated Settings (AI_RAG_* variables). Production secrets are supplied by
// the process environment only (see README "Environment").
module.exports = {
  apps: [
    {
      name: "landa-ai-rag",
      cwd: __dirname,
      script: "./.venv/Scripts/python.exe",
      args: "-m app",
      interpreter: "none",
      autorestart: true,
      min_uptime: 10000,
      exp_backoff_restart_delay: 5000,
      max_memory_restart: "768M",
      // Give uvicorn time to drain in-flight requests before PM2 force-kills.
      kill_timeout: 65000,
      env: {
        AI_RAG_ENV: "production",
        AI_RAG_HOST: "127.0.0.1",
        AI_RAG_PORT: "8010"
      },
      env_production: {
        AI_RAG_ENV: "production",
        AI_RAG_HOST: "127.0.0.1",
        AI_RAG_PORT: "8010"
      }
    }
  ]
}

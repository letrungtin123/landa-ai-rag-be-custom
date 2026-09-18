module.exports = {
  apps: [
    {
      name: "landa-ai-rag",
      cwd: __dirname,
      script: "./.venv/Scripts/python.exe",
      args: "-m uvicorn app.main:app --host 127.0.0.1 --port 8010",
      interpreter: "none",
      autorestart: true,
      min_uptime: 10000,
      exp_backoff_restart_delay: 5000,
      max_memory_restart: "768M",
      env: {
        NODE_ENV: "production",
        PORT: "8010"
      },
      env_production: {
        NODE_ENV: "production",
        PORT: "8010"
      }
    }
  ]
}
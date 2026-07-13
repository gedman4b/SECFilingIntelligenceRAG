#!/bin/bash
# Dev launcher: loads .env (ANTHROPIC_API_KEY) and starts the FastAPI app.
# Not part of the application; local convenience only.
set -a
source "$(dirname "$0")/.env"
set +a
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

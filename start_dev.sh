#!/usr/bin/env bash
# Start AfricaLens in development mode (both API + frontend)

# Load env vars if .env exists
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "Starting FastAPI server on :8000..."
uvicorn server.main:app --reload --port 8000 &
API_PID=$!

echo "Starting Vite dev server on :5173..."
cd client && npm run dev &
VITE_PID=$!

trap "kill $API_PID $VITE_PID 2>/dev/null" EXIT
wait

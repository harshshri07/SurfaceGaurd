#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-surfaceguard:latest}"
PORT="${PORT:-8501}"
USE_COMPOSE="${USE_COMPOSE:-0}"
DETACH="${DETACH:-1}"
NAME="${NAME:-surfaceguard-app}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${USE_COMPOSE}" == "1" ]]; then
  docker compose up --build
  exit $?
fi

docker build -t "${TAG}" .
docker rm -f "${NAME}" >/dev/null 2>&1 || true

if [[ "${DETACH}" == "1" ]]; then
  docker run -d --name "${NAME}" --restart unless-stopped -p "${PORT}:8501" "${TAG}"
  echo "Container started: ${NAME}"
  echo "URL: http://localhost:${PORT}"
  echo "Logs: docker logs -f ${NAME}"
  echo "Stop: docker rm -f ${NAME}"
else
  docker run --rm --name "${NAME}" -p "${PORT}:8501" "${TAG}"
fi

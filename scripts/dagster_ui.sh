#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAGSTER_HOME="${DAGSTER_HOME:-$repo_root/.dagster_home}"
mkdir -p "$DAGSTER_HOME"

cd "$repo_root"
echo "DAGSTER_HOME=$DAGSTER_HOME"
echo "Open the Dagster UI URL printed below, usually http://127.0.0.1:3000"
exec dagster dev -m quant_orchestrator.dagster_defs "$@"

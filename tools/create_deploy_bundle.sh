#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deploy_dir="$project_dir/deploy"
bundle_name="KO_deploy_$(date +%Y%m%d).tar.gz"
bundle_path="${1:-$deploy_dir/$bundle_name}"

mkdir -p "$(dirname "$bundle_path")"

# Package only reproducible runtime sources and required offline assets.
# setup.sh recreates Python environments and the force workspace build.
tar \
  --create \
  --gzip \
  --file "$bundle_path" \
  --directory "$project_dir" \
  --transform='s#^\.$#KO#;s#^\./#KO/#' \
  --exclude='./deploy' \
  --exclude='./.git' \
  --exclude='./.github' \
  --exclude='./.idea' \
  --exclude='./.vscode' \
  --exclude='./.venv' \
  --exclude='./ui/.venv' \
  --exclude='./ui/.env' \
  --exclude='./ui/instance' \
  --exclude='./output' \
  --exclude='./data/hit_records' \
  --exclude='./force_control/boxing_robot_ws/build' \
  --exclude='./force_control/boxing_robot_ws/install' \
  --exclude='./force_control/boxing_robot_ws/log' \
  --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.log' \
  --exclude='./.ko_final.lock' \
  .

archive_size="$(du -h "$bundle_path" | awk '{print $1}')"
checksum="$(sha256sum "$bundle_path" | awk '{print $1}')"
printf '[OK] deploy bundle: %s (%s)\n' "$bundle_path" "$archive_size"
printf '[OK] sha256: %s\n' "$checksum"
printf '[NEXT] tar -xzf %q && cd KO && ./setup.sh --build-force\n' "$bundle_path"

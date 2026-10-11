#!/usr/bin/env bash
# Project-local environment only; never modifies the source Python environment.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash scripts/setup_remote_env.sh /absolute/python /absolute/project/.env" >&2
  exit 2
fi
task_python=$1
task_prefix=$2
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "$task_prefix" != "$task_root/.env" || -e "$task_prefix" ]]; then
  echo "Refusing an existing environment or a prefix outside this project's .env" >&2
  exit 2
fi
"$task_python" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
"$task_python" -m venv "$task_prefix"
"$task_prefix/bin/python" -m pip install --upgrade pip
# CUDA 11.8 build selected for the existing server driver. Actual CUDA execution
# is verified separately; installing this wheel does not move QCR onto CUDA.
"$task_prefix/bin/python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
"$task_prefix/bin/python" -m pip install 'numpy>=1.24,<2'
"$task_prefix/bin/python" -m pip install -e "$task_root[dev]"
"$task_prefix/bin/python" -m pip check
"$task_prefix/bin/python" -m pip freeze

#!/bin/bash
# CLI wrapper for AISHELL-4 recording-level preparation.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${script_dir}/prepare_aishell4.py" "$@"

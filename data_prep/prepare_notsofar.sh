#!/bin/bash
# CLI wrapper for NotSoFar recording-level preparation.
#
# Example:
#   bash data_prep/prepare_notsofar.sh \
#     --output-root /data/nsf/recording \
#     --split train=/data/nsf/train_set/240825.1_train \
#     --split dev=/data/nsf/dev_set/240825.1_dev \
#     --split test=/data/nsf/test_set/240825.1_test

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${script_dir}/prepare_notsofar.py" --channels all "$@"

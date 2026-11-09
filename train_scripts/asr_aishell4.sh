#!/bin/bash
# Train AISHELL-4 spatial ASR.

set -euo pipefail
asr_root=""
dict_path=""
channel="0,1,2,3,4,5,6"
upstream="wavlm_base_plus"
cache_dir="downloads"
exp_dir=""
lr="0.0001"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --asr-root) asr_root="$2"; shift 2 ;;
        --dict) dict_path="$2"; shift 2 ;;
        --channel) channel="$2"; shift 2 ;;
        --upstream) upstream="$2"; shift 2 ;;
        --cache-dir) cache_dir="$2"; shift 2 ;;
        --exp-dir) exp_dir="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --asr-root PATH --dict PATH [--channel 0,1]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$asr_root" ]] || { echo "Missing --asr-root" >&2; exit 2; }
[[ -n "$dict_path" ]] || { echo "Missing --dict" >&2; exit 2; }
[[ -n "$exp_dir" ]] || exp_dir="exp/asr_aishell4/${upstream}_${lr}"
train_dir="${asr_root}/train_filter"
test_dir="${asr_root}/test"
[[ -f "${train_dir}/wav.scp" && -f "${test_dir}/wav.scp" ]] || {
    echo "Missing prepared AISHELL-4 train/test data" >&2
    exit 1
}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
extra_overrides="config.downstream_expert.datarc.channel='${channel}',,config.downstream_expert.datarc.dict_path='${dict_path}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir='${train_dir}',,config.downstream_expert.loaderrc.test_dir='${test_dir}'"
if [[ -d "${asr_root}/dev_filter" ]]; then
    extra_overrides="${extra_overrides},,config.downstream_expert.loaderrc.dev_dir='${asr_root}/dev_filter'"
fi
python3 "${repo_root}/run_downstream.py" \
    --cache_dir "$cache_dir" -p "$exp_dir" -m train -u "$upstream" \
    -d asr_aishell4 \
    -c "${repo_root}/downstream/asr_aishell4/config/AISHELL4/cfg.yaml" \
    -o "$extra_overrides"

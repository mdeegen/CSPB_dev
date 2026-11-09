#!/bin/bash
# Train the NotSoFar spatial ASR model.

set -euo pipefail

asr_root=""
dict_path=""
condition="MDM"
channel="0,1,2,3,4,5,6"
upstream="wavlm_base_plus"
cache_dir="downloads"
exp_dir=""
lr="0.0001"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --asr-root) asr_root="$2"; shift 2 ;;
        --dict) dict_path="$2"; shift 2 ;;
        --condition) condition="$2"; shift 2 ;;
        --channel) channel="$2"; shift 2 ;;
        --upstream) upstream="$2"; shift 2 ;;
        --cache-dir) cache_dir="$2"; shift 2 ;;
        --exp-dir) exp_dir="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --asr-root PATH --dict PATH [--condition MDM] [--channel 0,1]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$asr_root" ]] || { echo "Missing --asr-root" >&2; exit 2; }
[[ -n "$dict_path" ]] || { echo "Missing --dict" >&2; exit 2; }
[[ -n "$exp_dir" ]] || exp_dir="exp/asr_notsofar/${upstream}_${lr}_${condition}"

train_dir="${asr_root}/${condition}/train_filter"
dev_dir="${asr_root}/${condition}/dev_filter"
test_dir="${asr_root}/${condition}/test"
for directory in "$train_dir" "$dev_dir" "$test_dir"; do
    [[ -f "${directory}/wav.scp" && -f "${directory}/utt2num_samples" ]] || {
        echo "Missing prepared ASR data in ${directory}" >&2
        exit 1
    }
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python3 "${repo_root}/run_downstream.py" \
    --cache_dir "$cache_dir" \
    -p "$exp_dir" -m train -u "$upstream" -d asr_notsofar \
    -c "${repo_root}/downstream/asr_notsofar/config/NotSoFar/cfg.yaml" \
    -o "config.downstream_expert.datarc.channel='${channel}',,config.downstream_expert.datarc.dict_path='${dict_path}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir='${train_dir}',,config.downstream_expert.loaderrc.dev_dir='${dev_dir}',,config.downstream_expert.loaderrc.test_dir='${test_dir}'"

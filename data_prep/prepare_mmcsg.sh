#!/bin/bash
#
# prepare_mmcsg.sh — Prepare the MMCSG dataset for downstream ASR/diarization tasks.
#
# This script invokes prepare_mmcsg.py, which processes the raw MMCSG corpus
# (audio + transcriptions + RTTM diarization files) into Kaldi-style data directories
# for the dev, eval, and train splits. Each split directory will contain:
#   wav.scp    — mapping from recording ID to wav file path
#   utt2spk    — mapping from utterance ID to speaker ID
#   segments   — utterance-level time boundaries (utt_id, reco_id, start, end)
#   text       — utterance-level transcripts (lowercased)
#   reco2dur   — recording durations in seconds
#   uem        — un-partitioned evaluation map (full recording span)
#   rttm.scp   — mapping from recording ID to its RTTM diarization file
#
# Download the MMCSG dataset from: https://ai.meta.com/datasets/mmcsg-dataset/
# Expected structure under MMCSG_dir:
#   audio/{dev,eval,train}/*.wav
#   transcriptions/{dev,eval,train}/*.tsv   (columns: start end word speaker)
#   rttm/{dev,eval,train}/*.rttm

# Root directory of the raw MMCSG corpus
MMCSG_dir=/export/c02/hzili1/datasets/MMCSG/MMCSG

# Recording condition to extract:
#   SDM1 — Single Distant Microphone, channel 1 (mono; uses sox remix 1)
#   MDM  — Multiple Distant Microphones (all channels retained)
cond=SDM1

# Root output directory; split- and condition-specific subdirs will be created inside
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/MMCSG

python3 data_prep/prepare_mmcsg.py \
	${MMCSG_dir} \
	${output_dir} \
	--cond ${cond} \
	--merge_dis 0.5  # merge consecutive segments from the same speaker within 0.5 seconds

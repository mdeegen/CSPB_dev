#!/bin/bash
#
# prepare_dipco.sh — Prepare the DiPCo dataset for downstream ASR/diarization tasks.
#
# This script invokes prepare_dipco.py, which processes the raw DiPCo corpus
# (multi-channel audio + JSON transcription/diarization files) into Kaldi-style data
# directories for the dev and eval splits. Each split directory will contain:
#   wav.scp    — mapping from recording ID to wav file path
#   utt2spk    — mapping from utterance ID to speaker ID
#   segments   — utterance-level time boundaries (utt_id, reco_id, start, end)
#   text       — utterance-level transcripts (lowercased)
#   reco2dur   — recording durations in seconds
#   uem        — un-partitioned evaluation map (full recording span)
#   rttm.scp   — mapping from recording ID to its RTTM diarization file
#
# Recording IDs are formed as {session}_{mic} (e.g. S01_U01), where mic ∈ {U01..U05}.
# Each microphone unit has 7 channels (CH1–CH7); CH7 is used for SDM1.
# Transcriptions are read from per-session JSON files and optionally merged across
# consecutive same-speaker segments controlled by --merge_dis.
#
# Download the DiPCo dataset from: https://zenodo.org/record/8122551
# Expected structure under DiPCo_dir:
#   audio/{dev,eval}/{session}_{mic}.CH{1-7}.wav
#   transcriptions/{dev,eval}/{session}.json
#
# Output is written to: {output_dir}/{cond}/{dev,eval}/

# Root directory of the raw DiPCo corpus
DiPCo_dir=/export/corpora7/Dipco

# Recording condition to extract:
#   MDM  — Multiple Distant Microphones (all 7 channels merged via sox --combine merge)
#   SDM1 — Single Distant Microphone, channel 7 (CH7; no mixing)
cond=MDM

# Root output directory; condition- and split-specific subdirs will be created inside
output_dir=/export/fs05/hzili1/datasets/s3prl_csp/data/Dipco

python3 data_prep/prepare_dipco.py \
	${DiPCo_dir} \
	${output_dir} \
	--cond ${cond} \
	--merge_dis 0.0  # merge consecutive segments from the same speaker within 0.0 seconds (disabled)

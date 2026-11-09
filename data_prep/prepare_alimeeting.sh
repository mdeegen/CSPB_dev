#!/bin/bash
#
# prepare_alimeeting.sh — Prepare the AliMeeting dataset for downstream ASR/diarization tasks.
#
# This script invokes prepare_alimeeting.py, which processes the raw AliMeeting corpus
# (far-field audio + TextGrid transcription/diarization files) into Kaldi-style data
# directories for the Eval, Test, and Train splits. Each split directory will contain:
#   wav.scp    — mapping from recording ID to wav file path
#   utt2spk    — mapping from utterance ID to speaker ID
#   segments   — utterance-level time boundaries (utt_id, reco_id, start, end)
#   text       — utterance-level transcripts (M2MeT normalized)
#   reco2dur   — recording durations in seconds
#   uem        — un-partitioned evaluation map (full recording span)
#   rttm.scp   — mapping from recording ID to its RTTM diarization file
#
# Download the AliMeeting dataset from: https://www.openslr.org/119/
# Expected structure under Alimeeting_dir:
#   {Eval,Test,Train}_Ali/{Eval,Test,Train}_Ali_far/audio_dir/*.wav
#   {Eval,Test,Train}_Ali/{Eval,Test,Train}_Ali_far/textgrid_dir/*.TextGrid
#
# Output is written to: {output_dir}/{cond}/{Eval,Test,Train}/

# Root directory of the raw AliMeeting corpus
Alimeeting_dir=/export/c02/hzili1/datasets/Alimeeting

# Recording condition to extract:
#   SDM1 — Single Distant Microphone, channel 1 (mono; uses sox remix 1)
#   MDM8 — Multiple Distant Microphones, all 8 channels retained
cond=SDM1

# Root output directory; condition- and split-specific subdirs will be created inside
output_dir=/export/c02/hzili1/datasets/s3prl_csp/data/Alimeeting

python3 data_prep/prepare_alimeeting.py \
	${Alimeeting_dir} \
	${output_dir} \
	--cond ${cond}

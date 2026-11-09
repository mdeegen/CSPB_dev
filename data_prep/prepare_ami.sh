#!/bin/bash
#
# prepare_ami.sh — Prepare the AMI corpus for downstream ASR/diarization tasks.
#
# This script has two stages:
#
#   1. prepare_ami.py — Generates recording-level Kaldi files for train/dev/test splits:
#        wav.scp    — mapping from recording ID to wav file path
#        utt2spk    — recording-level speaker mapping (reco_id → reco_id)
#        rttm.scp   — mapping from recording ID to its RTTM diarization file
#        reco2dur   — recording durations in seconds
#        ref_rttm   — concatenated reference RTTM for all dev/test recordings (dev/test only)
#        uem        — concatenated UEM for all dev/test recordings (dev/test only)
#      RTTM and UEM files are taken from the BUT AMI diarization setup repo.
#
#   2. awk post-processing — Generates utterance-level files from ESPnet's pre-built
#      segment lists, reformatting utterance IDs from ESPnet's
#      {spk}_{corpus}_{session}_{start}_{end} scheme to {session}_{spk}_{start}_{end}:
#        segments   — utterance-level time boundaries (utt_id, reco_id, start, end)
#        utt2spk    — mapping from utterance ID to speaker ID
#        text       — utterance-level transcripts
#
# Dependencies:
#   BUT AMI diarization setup: https://github.com/BUTSpeechFIT/AMI-diarization-setup
#   ESPnet AMI recipe:         espnet/egs2/ami/asr1  (must be pre-built)
#
# Download the AMI corpus from: https://groups.inf.ed.ac.uk/ami/download/
# Expected audio structure under AMI_dir (for SDM1):
#   {meeting_id}/audio/{meeting_id}.Array1-01.wav
#
# Output is written to: {output_dir}/{train,dev,test}/


# Recording condition to extract:
#   SDM1    — Single Distant Microphone, Array1 channel 1 ({meeting_id}.Array1-01.wav)
#   IHM-MIX — Individual Headset Microphone mix ({meeting_id}.Mix-Headset.wav)
#   MDM8    — Multiple Distant Microphones, all 8 Array1 channels merged via sox
cond=SDM1

# Root directory of the raw AMI corpus
AMI_dir=/workspace/dataset/AMI

# Path to the ESPnet AMI recipe directory (must already have data/ populated)
espnet_dir=/workspace/workspace/espnet/egs2/ami/asr1

# Path to the cloned BUT AMI diarization setup repository
BUT_repo_dir=/workspace/workspace/AMI-diarization-setup

# Root output directory; split-specific subdirs will be created inside
output_dir=/workspace/dataset/CSPB/data/AMI/${cond}

# Stage 1: generate recording-level files (wav.scp, rttm.scp, reco2dur, ref_rttm, uem)
python3 data_prep/prepare_ami.py $BUT_repo_dir $AMI_dir $output_dir --cond $cond

# Stage 2: reformat ESPnet utterance-level files into output_dir
# ESPnet utterance ID format: {spk}_{corpus}_{session}_{ch}_{start}_{end}
# Reformatted to:             {session}_{spk}_{start}_{end}
# (fields a[2]=session, a[4]=start, a[5]=end, a[6]=end-of-segment; a[1]=spk)

# segments: utt_id → reco_id start end
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_train/segments > ${output_dir}/train/segments
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_dev/segments > ${output_dir}/dev/segments
awk '{split($1, a, "_"); print a[2]"_"a[4]"_"a[5]"_"a[6], a[2], $3, $4}' ${espnet_dir}/data/sdm1_eval/segments > ${output_dir}/test/segments

# utt2spk: utt_id → spk_id (first field of the reformatted utt_id)
awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/train/segments > ${output_dir}/train/utt2spk
awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/dev/segments > ${output_dir}/dev/utt2spk
awk -F' ' '{split($1, a, "_"); print $1, a[1]}' ${output_dir}/test/segments > ${output_dir}/test/utt2spk

# text: reformat utterance ID prefix, keep transcript unchanged
awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_train/text > ${output_dir}/train/text
awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_dev/text > ${output_dir}/dev/text
awk '{split($1, a, "_"); $1=a[2]"_"a[4]"_"a[5]"_"a[6]; print $0}' ${espnet_dir}/data/sdm1_eval/text > ${output_dir}/test/text

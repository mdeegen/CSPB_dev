#!/bin/bash
#
# beamformit.sh — Apply BeamformIt delay-and-sum beamforming to MMCSG MDM recordings.
#
# The MMCSG MDM condition produces 7-channel WAV files (channels 0–6). This script
# runs three separate beamforming experiments on those recordings, each using a
# different subset of channels, to allow downstream comparison of channel selections:
#
#   MDM_BF0,2      — 2 channels  (0, 2)
#   MDM_BF0,2,3,4  — 4 channels  (0, 2, 3, 4)
#   MDM_BF         — 7 channels  (0–6, all channels)
#
# Each experiment processes the dev, eval, and train splits. Beamformed mono WAV
# files are written to {mdm_bf_dir}/{split}/wav/. Channel indices are 0-indexed.
#
# BeamformIt configuration is read from data_prep/beamformit.cfg, which uses
# cross-correlation-based delay estimation with adaptive weights and automatic
# noise thresholding (originally adapted from the AMI recipe).
#
# Prerequisites:
#   - MMCSG MDM data directories must already exist under mdm_dir (run prepare_mmcsg.sh
#     with cond=MDM first).
#   - BeamformIt binary must be built; set BEAMFORMIT to its install directory.

source path.sh

# Source MDM data directory (produced by prepare_mmcsg.sh with cond=MDM)
mdm_dir=/export/c02/hzili1/datasets/s3prl_csp/data/MMCSG/MDM

# Number of parallel beamforming jobs per split
nj=16

# Experiment 1: beamform using channels 0 and 2 only
mdm_bf_dir=/export/fs05/hzili1/datasets/s3prl_csp/data/MMCSG/MDM_BF0,2

for dset in eval dev train; do
  python3 data_prep/beamformit_kaldi.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2
done

# Experiment 2: beamform using channels 0, 2, 3, and 4
mdm_bf_dir=/export/fs05/hzili1/datasets/s3prl_csp/data/MMCSG/MDM_BF0,2,3,4

for dset in eval dev train; do
  python3 data_prep/beamformit_kaldi.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,2,3,4
done

# Experiment 3: beamform using all 7 channels (0–6)
mdm_bf_dir=/export/fs05/hzili1/datasets/s3prl_csp/data/MMCSG/MDM_BF

for dset in eval dev train; do
  python3 data_prep/beamformit_kaldi.py ${mdm_dir}/${dset}/wav.scp ${mdm_bf_dir}/${dset} --config_file data_prep/beamformit.cfg --num_jobs ${nj} --channels 0,1,2,3,4,5,6
done

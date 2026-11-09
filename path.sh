#!/bin/bash
#
# path.sh — Environment setup for data_prep scripts.
#
# Source this file at the top of any data_prep script that needs a specific
# Python environment or external tool on PATH:
#
#   source path.sh
source /net/vol/deegen/cspb/.cspb/bin/activate

# Python environment (anaconda): contains soundfile, joblib, tqdm, textgrid, etc.
#export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"

# BeamformIt binary (required by beamformit.sh / beamformit_kaldi.py)
#export BEAMFORMIT=/export/c02/hzili1/workspace/espnet/tools/BeamformIt
#export PATH="${PATH}:${BEAMFORMIT}"

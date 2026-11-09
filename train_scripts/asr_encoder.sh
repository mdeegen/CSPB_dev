#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15360
#SBATCH --job-name=asr_ami
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

source path.sh

# Directory where s3prl caches upstream model weights
#cache_dir=/path/to/s3prl/downloads
cache_dir=/net/vol/ameise/cache/s3prl/

# Upstream model. Use any s3prl-supported name (e.g. hubert_base, wavlm_base_plus).
# To use a custom pre-trained upstream, uncomment and set:
#   upstream=custom_local
#   ckpt=/path/to/upstream/checkpoint
#   and add:  -k ${ckpt}  to the run_downstream.py call below.
upstream=wavlm_base_plus

lr=0.0001
gpus=1
port=25652
#distributed="-m torch.distributed.launch --nproc_per_node ${gpus} --master_port ${port}"
distributed=""

# Recording condition. Controls which data directory and channel(s) to use.
# Set cond to one of the values handled in the if-block below.
cond="mdm_0,2,4,6"
if [[ "$cond" == "mdm_0,2,4,6" ]]; then
    data="MDM"
    channel="0,2,4,6"
elif [[ "$cond" == "mdm_0,4" ]]; then
    data="MDM"
    channel="0,4"
elif [[ "$cond" == "mdm_bf0,2,4,6" ]]; then
    data="MDM_BF0,2,4,6"
    channel="0"
elif [[ "$cond" == "mdm_bf0,4" ]]; then
    data="MDM_BF0,4"
    channel="0"
elif [[ "$cond" == "mdm_bfall" ]]; then
    data="MDM_BF"
    channel="0"
elif [[ "$cond" == "sdm1" ]]; then
    data="SDM1"
    channel="0"
else
    exit 1
fi

#exp_dir="`pwd`/exp/asr_ami/${upstream}_${lr}_${cond}"
#
## Root directory containing the segmented ASR data produced by
## downstream/asr_ami/prepare_asr_seg.sh. Expected sub-directories:
##   ${data}/train_filter, ${data}/dev_filter, ${data}/test
#asr_data_dir=/path/to/downstream/asr_ami
#train_dir=${asr_data_dir}/${data}/train_filter
#dev_dir=${asr_data_dir}/${data}/dev_filter
#test_dir=${asr_data_dir}/$wavlm_base_plus{data}/test

exp_dir="/net/vol/deegen/cspb/exp/asr_enc/${upstream}_${lr}_${cond}"
mkdir -p $exp_dir

#Root directory containing the segmented ASR data produced by
#downstream/asr_ami/prepare_asr_seg.sh. Expected sub-directories:
# TODO: Fix to all Datasets
asr_data_dir="/net/vol/ameise/data/CSPB/AMI/asr_mdm8"       # DATEN DIFFERENT
train_dir=${asr_data_dir}/train_filter
dev_dir=${asr_data_dir}/dev_filter
test_dir=${asr_data_dir}/test

echo $train_dir
echo $dev_dir
echo $test_dir
echo $channel
echo $exp_dir

python3 $distributed run_downstream.py \
    --cache_dir ${cache_dir} \
    -p $exp_dir \
    -m train \
    -u $upstream \
    -d asr_ami \
    -c downstream/asr_ami/config/AMI/cfg_enc.yaml \
    -o "config.downstream_expert.datarc.channel='${channel}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir=${train_dir},,config.downstream_expert.loaderrc.dev_dir=${dev_dir},,config.downstream_expert.loaderrc.test_dir=${test_dir}"


    # CONFIG DIFFERENT
# TODOS:
# ---- DONE
# Find model point and exchange for our spatial encoder 
# Fix model forward pass with new model und trianable aprams but leave rest for metrics, loggign etc
# Find LSTM module and adapt output of encoder to fit expected output as lstm was
# make sure mdoel can use function model.parameters()
# GCC FEAtures
# check params like learning rate etc for encoder setup,


# TODOs Open
# Figure out how to train on all datasets simultaneously (e.g. AMI, CHiME-6, etc.) and how to adapt the output of the encoder to fit the expected output of the LSTM
# increase max iterations
# METRIC Training bei uer UND wer weil alle datensätze, wie amcht man das bei gemsichten datensätzen?

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15360
#SBATCH --job-name=diar_mmcsg
#SBATCH --time=5-00:00:00
#SBATCH --gpus=1

source path.sh

# Directory where s3prl caches upstream model weights
cache_dir=/path/to/s3prl/downloads

# Upstream model. Use any s3prl-supported name (e.g. hubert_base, wavlm_base_plus).
# To use a custom pre-trained upstream, uncomment and set:
#   upstream=custom_local
#   ckpt=/path/to/upstream/checkpoint
#   and add:  -k ${ckpt}  to the run_downstream.py call below.
upstream=wavlm_base_plus

gpus=1
port=25652
#distributed="-m torch.distributed.launch --nproc_per_node ${gpus} --master_port ${port}"
distributed=""

# Recording condition. Controls which data directory and channel(s) to use.
# Set cond to one of the values handled in the if-block below.
cond=sdm1
if [[ "$cond" == "mdm_0,2,3,4" ]]; then
    data="MDM"
    channel="0,2,3,4"
elif [[ "$cond" == "mdm_0,2" ]]; then
    data="MDM"
    channel="0,2"
elif [[ "$cond" == "mdm_bf0,2,3,4" ]]; then
    data="MDM_BF0,2,3,4"
    channel="0"
elif [[ "$cond" == "mdm_bf0,2" ]]; then
    data="MDM_BF0,2"
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

# Root directory containing the diarization data produced by data_prep/prepare_mmcsg.sh.
# Expected sub-directories: ${data}/train_filter, ${data}/dev_filter
diar_data_dir=/path/to/downstream/diar_mmcsg
train_dir=${diar_data_dir}/${data}/train_filter
dev_dir=${diar_data_dir}/${data}/dev_filter

# Run training for each learning rate (hyperparameter search).
for lr in 0.001 0.0001 0.00001; do
  exp_dir="`pwd`/exp/diar_mmcsg/${upstream}_${lr}_${cond}"
  echo ${train_dir}
  echo ${dev_dir}
  echo ${channel}
  echo ${exp_dir}

  python3 $distributed run_downstream.py \
      --cache_dir ${cache_dir} \
      -p $exp_dir \
      -m train \
      -u $upstream \
      -d diar_mmcsg \
      -c downstream/diar_mmcsg/config/MMCSG/cfg.yaml \
      -o "config.downstream_expert.datarc.channel='${channel}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir=${train_dir},,config.downstream_expert.loaderrc.dev_dir=${dev_dir}"
done

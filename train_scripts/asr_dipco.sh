#!/bin/bash


source path.sh

# Directory where s3prl caches upstream model weights
cache_dir=/net/vol/ameise/cache/s3prl/


# Upstream model. Use any s3prl-supported name (e.g. hubert_base, wavlm_base_plus).
# To use a custom pre-trained upstream, uncomment and set:
#   upstream=custom_local
#   ckpt=/path/to/upstream/checkpoint
#   and add:  -k ${ckpt}  to the run_downstream.py call below.
upstream=wavlm_base_plus

lr=0.00001    $1 #0.00003 #0.00005 0.000003
echo "using lr: $lr"
gpus=1

#distributed="-m torch.distributed.launch --nproc_per_node ${gpus} --master_port ${port}"
distributed=""

# Recording condition. Controls which data directory and channel(s) to use.
# Set cond to one of the values handled in the if-block below.
cond="mdm_bf0,2,4,6"
channel="0"
data="MDM_BF0,2,4,6"




#exp_dir="/net/vol/ameise/project_storage/CSPB/asr_dipco/${upstream}_${lr}_${cond}"
exp_dir="/net/vol/deegen/cspb/exp/asr_dipco/${upstream}_${lr}_${cond}"

# Root directory containing the segmented ASR data produced by
# downstream/asr_dipco/prepare_asr_seg.sh. Expected sub-directories:
#   ${data}/train_filter, ${data}/dev_filter, ${data}/test
asr_data_dir="/net/vol/ameise/data/CSPB/DiPCo/asr_bfit"
train_dir=${asr_data_dir}/dev_filter
dev_dir=${asr_data_dir}/eval_filter
test_dir=${asr_data_dir}/eval

echo $train_dir
echo $dev_dir
echo $test_dir
echo $channel
echo $exp_dir


init_ckpt="/net/vol/deegen/cspb/exp/asr_ami/wavlm_base_plus0.0001mdm_bf0,2,4,6/dev-best.ckpt"
#"/net/vol/ameise/project_storage/CSPB/asr_ami/wavlm_base_plus_0.0001_mdm_bf0,2,4,6/dev-best.ckpt"
echo -e "using init ckpt dir:\n "
echo $init_ckpt
echo -e "\n"

python3 $distributed run_downstream.py \
    --cache_dir ${cache_dir} \
    -p $exp_dir \
    -m train \
    -u $upstream \
    -d asr_dipco \
    -c downstream/asr_dipco/config/DiPCo_final/cfg.yaml \
    -o "config.downstream_expert.datarc.channel='${channel}',,config.optimizer.lr=${lr},,config.downstream_expert.loaderrc.train_dir=${train_dir},,config.downstream_expert.loaderrc.dev_dir=${dev_dir},,config.downstream_expert.loaderrc.test_dir=${test_dir}" \
    --init_ckpt $init_ckpt
    #--past_exp overwrites config cf. run_downstrem l. 34


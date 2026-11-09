#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=asr_alm_infer
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --exclude=c04

source path.sh
export PATH="/net/vol/deegen/cspb/.cspb/bin:$PATH"

gpus=1
port=25678

cond="mdm_bf0,2,4,6"
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
    exit 1;
fi

#exp_dir="/export/c02/hzili1/workspace/s3prl/s3prl/exp/asr_alimeeting/"
#ckpt="${exp_dir}/dev-best.ckpt"
#test_dir="/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_alimeeting/${data}/Test"

exp_dir="/net/vol/deegen/cspb/exp/asr_alimeeting/wavlm_base_plus_0.0001_mdm_bf0,2,4,6"
ckpt="${exp_dir}/dev-best.ckpt"
test_dir="/net/vol/ameise/data/CSPB/ALM/asr_bfit/Test"

python3 run_downstream.py \
    -m evaluate \
    -e $ckpt \
    -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='${channel}'"

./downstream/asr_alimeeting/score.sh $exp_dir false $test_dir

#python3 run_downstream.py \
#	-m evaluate \
#	-e $ckpt \
#	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"
#
#./downstream/asr_ami/score.sh $exp_dir true $test_dir

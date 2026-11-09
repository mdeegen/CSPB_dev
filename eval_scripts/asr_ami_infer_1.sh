#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=asr_ami
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --exclude=c04

export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"

gpus=1
port=25678

exp_dir="/export/c02/hzili1/workspace/s3prl/s3prl/exp/asr_ami_past/wavlm_base_0.0001"
ckpt="${exp_dir}/dev-best_new.ckpt"
test_dir="/export/c02/hzili1/datasets/s3prl_csp/downstream/asr_ami/SDM1/test"

python3 run_downstream.py \
    -m evaluate \
    -e $ckpt \
    -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='0'"

#./downstream/asr_ami/score.sh $exp_dir false $test_dir

#python3 run_downstream.py \
#	-m evaluate \
#	-e $ckpt \
#	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"
#
#./downstream/asr_ami/score.sh $exp_dir true $test_dir

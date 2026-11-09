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

export PATH="/net/vol/deegen/cspb/.cspb/bin:$PATH"

gpus=1
port=25678

exp_dir="/net/vol/deegen/cspb/exp/asr_ami/wavlm_base_plus0.0001mdm_bf0,2,4,6"
ckpt="${exp_dir}/dev-best.ckpt"
test_dir="/net/vol/ameise/data/CSPB/AMI/asr_bfit/test"

python3 run_downstream.py \
    -m evaluate \
    -e $ckpt \
    -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='0'"

./downstream/asr_ami/score.sh $exp_dir false $test_dir

#python3 run_downstream.py \
#	-m evaluate \
#	-e $ckpt \
#	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"
#
#./downstream/asr_ami/score.sh $exp_dir true $test_dir

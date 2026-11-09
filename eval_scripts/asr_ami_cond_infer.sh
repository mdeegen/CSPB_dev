#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --job-name=asr_ami_eval
#SBATCH --time=10:00:00
# --gpus=1
if [[ -z "$1" ]]; then
    echo "ERROR: No experiment directory provided!" >&2
    echo "Usage: $0 <exp_dir>" >&2
    exit 1
fi
exp_dir=$1
echo "exp_dir: $exp_dir"
# export PATH="/net/vol/deegen/cspb/.cspb/bin:$PATH"
source path.sh

gpus=1
port=25678

# exp_dir="/net/vol/deegen/cspb/exp/asr_enc_alt/wavlm_base_plus_0.0001_mdm_0,2,4,6_wrong_dsetfile"
ckpt="${exp_dir}/dev-best.ckpt"
channel="0,2,4,6"

test_dir="/scratch/hpc-prf-nt2/deegen/data/CSPB/AMI/asr_mdm8/test"

# python3 run_downstream.py \
#     -m evaluate \
#     -e $ckpt \
#     -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='${channel}'"
# # todo: !!!!!!!!!TODOD: MAN MUSSTE CHANNELS HIER ANPASSEN SONST NUR SINGLE CHANNEL!!!!!

./downstream/asr_ami/score_n2.sh $exp_dir false $test_dir

#python3 run_downstream.py \
#	-m evaluate \
#	-e $ckpt \
#	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"
#
#./downstream/asr_ami/score.sh $exp_dir true $test_dir

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
if [[ -z "$1" ]]; then
    echo "ERROR: No experiment directory provided!" >&2
    echo "Usage: $0 <exp_dir>" >&2
    exit 1
fi
exp_dir=$1
echo "exp_dir: $exp_dir"
export PATH="/net/vol/deegen/cspb/.cspb/bin:$PATH"

gpus=1
port=25678

exp_dir=$exp_dir # "/net/vol/deegen/cspb/exp/asr_enc_alt/wavlm_base_plus_0.0001_mdm_0,2,4,6_wrong_dsetfile"
ckpt="${exp_dir}/dev-best.ckpt"
channel="0,2,4,6"

test_dir="/net/vol/ameise/data/CSPB/mmcsg/asr_mdm/eval"

python3 run_downstream.py \
    -m evaluate \
    -e $ckpt \
    -o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.channel='${channel}'"
# todo: !!!!!!!!!TODOD: MAN MUSSTE CHANNELS HIER ANPASSEN SONST NUR SINGLE CHANNEL!!!!!
./downstream/asr_mmcsg/score.sh $exp_dir false $test_dir

#python3 run_downstream.py \
#	-m evaluate \
#	-e $ckpt \
#	-o "config.downstream_expert.datarc.max_samples=1000000,,config.downstream_expert.loaderrc.eval_batchsize=1,,config.downstream_expert.loaderrc.test_dir=${test_dir},,config.downstream_expert.datarc.mch=False,,config.downstream_expert.datarc.channel='0',,config.downstream_expert.datarc.decoder_args.decoder_type='kenlm'"
#
#./downstream/asr_ami/score.sh $exp_dir true $test_dir

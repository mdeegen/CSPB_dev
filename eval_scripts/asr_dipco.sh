#!/bin/bash

export PATH="/net/vol/deegen/cspb/.cspb/bin:$PATH"

gpus=1


#exp_dir="/net/vol/ameise/project_storage/CSPB/asr_dipco/wavlm_base_plus_0.00001_mdm_bf0,2,4,6"
#ckpt="${exp_dir}/states-103000.ckpt"
#test_dir="/net/vol/ameise/data/CSPB/DiPCo/asr_bfit/eval"

exp_dir="/net/vol/deegen/cspb/exp/asr_dipco/wavlm_base_plus_0.00003_mdm_bf0,2,4,6"
ckpt="${exp_dir}/states-103000.ckpt"
test_dir="/net/vol/ameise/data/CSPB/DiPCo/asr_bfit/eval"  # ohne filter? aber fehler input zu kurz


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

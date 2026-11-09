#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=diar_eval
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --exclude=c04,octopod

source path.sh

gpus=1
port=25652
normalize=1

exp_dir="exp/diar_mmcsg/"
ckpt="${exp_dir}/best-states-dev.ckpt"

cond=sdm1
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
elif [[ "$cond" == "sdm1" ]]; then
    data="SDM1"
    channel="0"
else
    exit 1;
fi

data_dir=/export/c02/hzili1/datasets/s3prl_csp/data/MMCSG/${data}
dev_dir="${data_dir}/dev"
test_dir="${data_dir}/eval"

echo $dev_dir
echo $test_dir
echo $channel
echo $ckpt

min_cluster_size=15
cluster_thres=
segmentation_thres=0.5

best_der=100
best_threshold=0

for cluster_thres in 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7; do
  echo "Cluster threshold $cluster_thres"
  output_dir=$exp_dir/rttm/dev
  mkdir -p $output_dir

  python3 downstream/diar_ami/evaluate_v1.py \
  	$ckpt \
  	$dev_dir \
  	$output_dir \
	--channel $channel \
  	--normalize $normalize \
  	--min_cluster_size $min_cluster_size \
  	--cluster_thres $cluster_thres \
  	--segmentation_thres $segmentation_thres

  cat $output_dir/*.rttm > $output_dir/hyp_rttm

  output=$(./downstream/diar_ami/md-eval.pl -r ${dev_dir}/ref_rttm -s $output_dir/hyp_rttm -u ${dev_dir}/uem | python3 downstream/diar_ami/parse_md_eval_output.py)
  IFS=' ' read -r der MISS FA CF <<< "$output"
  echo "DER $der, MISS+FA+CF=$MISS+$FA+$CF"
  if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
      best_der=$der
      best_threshold=$cluster_thres
  fi
done

echo "best der $best_der, best threshold $best_threshold"

output_dir=$exp_dir/rttm/test
mkdir -p $output_dir
echo $best_threshold > $output_dir/best_threshold

python3 downstream/diar_ami/evaluate_v1.py \
	$ckpt \
	$test_dir \
	$output_dir \
	--channel $channel \
	--normalize $normalize \
	--min_cluster_size $min_cluster_size \
	--cluster_thres $best_threshold \
	--segmentation_thres $segmentation_thres
cat $output_dir/*.rttm > $output_dir/hyp_rttm
./downstream/diar_ami/md-eval.pl -r ${test_dir}/ref_rttm -s $output_dir/hyp_rttm -u ${test_dir}/uem

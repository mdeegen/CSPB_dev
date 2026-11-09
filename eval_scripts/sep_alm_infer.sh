#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15360
#SBATCH --job-name=sep_infer
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1
#SBATCH --exclude=c04,octopod,c08,c14

source path.sh

normalize=1

exp_dir="/export/c02/hzili1/workspace/s3prl/s3prl/exp/sep_alm/wavlm_base_plus_0.0001_sdm1"
ckpt="${exp_dir}/best-states-dev.ckpt"

data_dir="/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_alimeeting/SDM1/"
sdm1_data_dir="/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_alimeeting/SDM1/"
test_sets="Test_utt_group_1spk Test_utt_group_2spk"
channel='0'

AMI_dir=/export/c02/hzili1/workspace/espnet/egs2/AliMeeting/asr/
asr_exp=asr_train_asr_conformer
asr_config=$AMI_dir/exp/$asr_exp/config.yaml
asr_model=$AMI_dir/exp/$asr_exp/valid.acc.ave.pth
lm_exp=lm_train_lm_transformer
lm_config=$AMI_dir/exp/$lm_exp/config.yaml
lm_model=$AMI_dir/exp/$lm_exp/valid.loss.ave_10best.pth
decode_config=$AMI_dir/conf/decode_asr_rnn.yaml
nj=32
echo $exp_dir
echo $asr_exp

start_time=$(date +%s)

for dir in $test_sets; do
  test_dir="$data_dir/$dir"
  sdm1_dir="$sdm1_data_dir/$dir"
  output_dir=$exp_dir/infer/$dir
  num_srcs=$(cat $test_dir/num_srcs)
  _logdir="${output_dir}/logdir"

  #${AMI_dir}/utils/slurm.pl --config $AMI_dir/conf/slurm.conf --gpu 1 "${_logdir}"/infer_sep.log \
  
  python3 downstream/sep_alimeeting/infer.py $ckpt $test_dir $sdm1_dir $output_dir --channel $channel --normalize $normalize --num_srcs $num_srcs
done

current_dir=$(pwd)

for dir in $test_sets; do
  test_dir="$data_dir/$dir"
  output_dir=$exp_dir/infer/$dir
  mkdir -p $output_dir/split${nj}
  key_file=${output_dir}/wav.scp
  split_scps=""
  for n in $(seq "${nj}"); do
      split_scps+=" ${output_dir}/split${nj}/keys.${n}.scp"
  done

  cd $AMI_dir
  source path.sh

  # shellcheck disable=SC2046,SC2086
  utils/split_scp.pl "${key_file}" ${split_scps}


  # 2. Submit decoding jobs
  _logdir="${output_dir}/logdir"
  rm -f "${_logdir}/*.log"
  # shellcheck disable=SC2046,SC2086
  utils/slurm.pl --config conf/slurm.conf --gpu 0 JOB=1:"${nj}" "${_logdir}"/asr_inference.JOB.log \
      python3 -m espnet2.bin.asr_inference \
          --batch_size 1 \
          --ngpu 0 \
          --data_path_and_name_and_type "${output_dir}/wav.scp,speech,sound" \
          --key_file $output_dir/split${nj}/keys.JOB.scp \
          --asr_train_config $asr_config \
          --asr_model_file $asr_model \
          --output_dir "${_logdir}"/output.JOB \
          --config ${decode_config} \
          --lm_train_config ${lm_config} \
          --lm_file ${lm_model} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; }

  for f in token token_int score text; do
      if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
          for i in $(seq "${nj}"); do
              cat "${_logdir}/output.${i}/1best_recog/${f}"
          done | sort -k1 >"${output_dir}/${f}"
      fi
  done

  cd $current_dir
  source path.sh
  python3 downstream/sep_alimeeting/permute.py $test_dir/text ${output_dir}/text ${output_dir} --token char
  ./downstream/sep_alimeeting/score.sh $output_dir char 
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

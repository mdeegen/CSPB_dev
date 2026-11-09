#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15360
#SBATCH --job-name=sep_infer
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"
export PYTHONPATH="/export/c02/hzili1/workspace/s3prl/s3prl:$PYTHONPATH"

normalize=1

exp_dir="/export/c02/hzili1/workspace/s3prl/s3prl/exp/sep_ami/"
ckpt="${exp_dir}/best-states-dev_new.ckpt"

data_dir="/export/c02/hzili1/datasets/s3prl_csp/downstream/sep_ami/SDM1/"
test_sets="test_utt_group_1spk test_utt_group_2spk"

AMI_dir=/export/c02/hzili1/workspace/espnet/egs2/ami/asr1
asr_exp=asr_train_asr_conformer_raw_en_bpe100_sp
asr_config=$AMI_dir/exp/$asr_exp/config.yaml
asr_model=$AMI_dir/exp/$asr_exp/valid.acc.ave.pth
lm_config=$AMI_dir/exp/lm_train_lm_transformer2_en_bpe100/config.yaml
lm_model=$AMI_dir/exp/lm_train_lm_transformer2_en_bpe100/valid.loss.ave_10best.pth
decode_config=$AMI_dir/conf/tuning/decode_transformer2.yaml
nj=32
echo $exp_dir
echo $asr_exp

start_time=$(date +%s)

for dir in $test_sets; do
  test_dir="$data_dir/$dir"
  output_dir=$exp_dir/infer/$dir
  num_srcs=$(cat $test_dir/num_srcs)
  _logdir="${output_dir}/logdir"

  #${AMI_dir}/utils/slurm.pl --config $AMI_dir/conf/slurm.conf --gpu 1 "${_logdir}"/infer_sep.log \
  
  python3 downstream/sep_ami/infer.py $ckpt $test_dir $output_dir --normalize $normalize --num_srcs $num_srcs
done

for dir in $test_sets; do
  test_dir="$data_dir/$dir"
  output_dir=$exp_dir/infer/$dir
  mkdir -p $output_dir/split${nj}
  key_file=${output_dir}/wav.scp
  split_scps=""
  for n in $(seq "${nj}"); do
      split_scps+=" ${output_dir}/split${nj}/keys.${n}.scp"
  done
  # shellcheck disable=SC2046,SC2086
  ${AMI_dir}/utils/split_scp.pl "${key_file}" ${split_scps}

  source $AMI_dir/path.sh

  # 2. Submit decoding jobs
  _logdir="${output_dir}/logdir"
  rm -f "${_logdir}/*.log"
  # shellcheck disable=SC2046,SC2086
  ${AMI_dir}/utils/slurm.pl --config $AMI_dir/conf/slurm.conf --gpu 0 JOB=1:"${nj}" "${_logdir}"/asr_inference.JOB.log \
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

  export PATH="/export/c02/hzili1/tmp/home/hzili1/anaconda3/envs/csp/bin:$PATH"
  python3 downstream/sep_ami/permute.py $test_dir/text ${output_dir}/text ${output_dir}
  ./downstream/sep_ami/score.sh $output_dir word 
done

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

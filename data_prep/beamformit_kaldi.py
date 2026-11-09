"""
beamformit_kaldi.py — Run BeamformIt beamforming on a Kaldi wav.scp file.

Reads a Kaldi-format wav.scp (recording_id  wav_path), extracts the requested
channels from each multi-channel recording, runs BeamformIt to produce a single
beamformed WAV, and writes the result to {output_dir}/wav/{recording_id}.wav.

Supports shard-based parallelism for multi-node/multi-process use: the input
list is divided into --split shards and this process handles shard --rank.
Within a shard, recordings are processed in parallel using joblib.

BeamformIt requires a minimum of 16000 samples; recordings shorter than this are
zero-padded before beamforming and trimmed back to their original length afterward.

Temporary per-channel WAV files are written to {output_dir}/tmp_{split}_{rank}/
and removed after each recording is processed. BeamformIt side-car files
(.del, .del2, .info, .weat) are also cleaned up.

Usage:
  python3 beamformit_kaldi.py <wav_scp_file> <output_dir> \\
      --config_file <beamformit.cfg> \\
      [--channels 0,1,2] [--split N] [--rank K] [--num_jobs J]
"""

import os
import time
import argparse
import subprocess
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Beamformit')
parser.add_argument('wav_scp_file', type=str, help='wav scp file')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--config_file', type=str, help='BeamformIt configuration file')
parser.add_argument('--split', type=int, default=1, help='Total number of shards (for distributed processing)')
parser.add_argument('--rank', type=int, default=0, help='Index of the shard this process handles (0-indexed)')
parser.add_argument('--num_jobs', type=int, default=8, help='Number of parallel jobs within this shard')
parser.add_argument('--channels', type=str, default='', help='Comma-separated 0-indexed channel numbers to use (default: all channels)')
args = parser.parse_args()

def batch_process(lines, num_jobs):
    """Process a list of wav.scp lines in parallel using joblib.

    Args:
        lines:    List of wav.scp lines (recording_id  wav_path).
        num_jobs: Number of parallel workers.

    Returns:
        List of return codes from beamform() (all 0 on success).
    """
    results = Parallel(n_jobs=num_jobs)(
        delayed(beamform)(line)
        for line in tqdm(lines)
    )
    return results

def pad_speech(speech_path, nsamples):
    """Zero-pad a WAV file to at least nsamples frames, in-place.

    BeamformIt requires a minimum number of samples to operate. Short recordings
    are padded with trailing zeros before beamforming and trimmed back afterward
    by trim_speech.

    Args:
        speech_path: Path to the WAV file to pad.
        nsamples:    Target minimum number of samples.
    """
    speech, sr = sf.read(speech_path)
    if len(speech) < nsamples:
        padded_speech = np.zeros((nsamples, ))
        padded_speech[:len(speech)] = speech
    else:
        padded_speech = speech
    sf.write(speech_path, padded_speech, sr)
    return 0

def trim_speech(speech_path, nsamples):
    """Trim a WAV file to exactly nsamples frames, in-place.

    Used after beamforming to restore the original length of recordings that
    were padded by pad_speech.

    Args:
        speech_path: Path to the WAV file to trim.
        nsamples:    Number of samples to retain from the start of the file.
    """
    speech, sr = sf.read(speech_path)
    sf.write(speech_path, speech[:nsamples], sr)
    return 0

def beamform(line):
    """Beamform a single recording from a wav.scp line.

    Steps:
      1. Extract each requested channel as a separate mono WAV in the tmp dir
         using sox remix (channels are 1-indexed in sox, so 0-indexed args are
         incremented by 1).
      2. Pad channels shorter than 16000 samples to satisfy BeamformIt's minimum.
      3. Write a BeamformIt channel list file and invoke BeamformIt.
      4. Trim the beamformed output back to the original sample count if padded.
      5. Remove temporary channel files and BeamformIt side-car outputs.

    Args:
        line: A single wav.scp line: "recording_id  /path/to/multichannel.wav"
    """
    line = line.strip('\n')
    line_split = line.split()
    uttname, wav_path = line_split[0], line_split[1]
    assert os.path.exists(wav_path)
    channel_str = "{}".format(uttname)
    num_samples = (sf.info(wav_path)).frames

    if args.channels != '':
        nchans = [int(c) for c in args.channels.split(',')]
    else:
        nchans = list(range(sf.info(wav_path).channels))

    min_samples = 16000
    # temporary store selected channels
    for c in nchans:
        cmd = "sox {} {}/tmp_{}_{}/{}_{}.wav remix {}".format(wav_path, args.output_dir, args.split, args.rank, uttname, c, c+1)
        status, output = subprocess.getstatusoutput(cmd)
        assert status == 0
        if num_samples < min_samples:
            pad_speech("{}/tmp_{}_{}/{}_{}.wav".format(args.output_dir, args.split, args.rank, uttname, c), min_samples)
        channel_str += " " + "{}_{}.wav".format(uttname, c) 
    channel_str += '\n'
    channel_file = "{}/tmp_{}_{}/channels_{}".format(args.output_dir, args.split, args.rank, uttname)
    with open(channel_file, 'w') as fh:
        fh.write(channel_str)
    cmd = "BeamformIt -s {} -c {} --config_file {} --source_dir {}/tmp_{}_{} --result_dir {}/wav".format(uttname, channel_file, args.config_file, args.output_dir, args.split, args.rank, args.output_dir)
    status, output = subprocess.getstatusoutput(cmd)
    assert status == 0
    
    if num_samples < min_samples:
        trim_speech("{}/wav/{}.wav".format(args.output_dir, uttname), num_samples)

    # remove extra files
    for c in nchans:
        os.remove("{}/tmp_{}_{}/{}_{}.wav".format(args.output_dir, args.split, args.rank, uttname, c))
    for ext in ["del", "del2", "info", "weat"]:
        os.remove("{}/wav/{}.{}".format(args.output_dir, uttname, ext))
    return 0

def main():
    """Entry point: read wav.scp, select this shard, and run beamforming in parallel.

    Sharding: line i belongs to shard (i % split). This process handles lines
    where (i % split) == rank, allowing N independent processes to cover all
    recordings without overlap.
    """
    print(args)
    if not os.path.exists(args.output_dir + '/tmp_{}_{}'.format(args.split, args.rank)):
        os.makedirs(args.output_dir + '/tmp_{}_{}'.format(args.split, args.rank))
    if not os.path.exists(args.output_dir + '/wav'):
        os.makedirs(args.output_dir + '/wav')
    with open(args.wav_scp_file, 'r') as fh:
        content = fh.readlines()
    cnt = 0
    start_time = time.time()
    content_rank = [line for i, line in enumerate(content) if i % args.split == args.rank]
    print("Rank {}, process {} / {} utts".format(args.rank, len(content_rank), len(content)))

    batch_process(content_rank, num_jobs=args.num_jobs)
    #for line in tqdm(content_rank):
    #    beamform(line)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    return 0

if __name__ == '__main__':
    main()

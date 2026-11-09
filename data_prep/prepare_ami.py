"""
prepare_ami.py — Prepare the AMI corpus for downstream ASR/diarization tasks.

Generates recording-level Kaldi-style data directories for the train, dev, and
test splits. RTTM and UEM files are sourced from the BUT AMI diarization setup
repository; transcription segments must be added separately from an ESPnet recipe
(see prepare_ami.sh for the awk post-processing step).

Each output split directory (under {output_dir}/{split}/) contains:
  wav.scp   — recording ID → wav path (or sox pipe for MDM8)
  utt2spk   — recording ID → recording ID (recording-level placeholder)
  rttm.scp  — recording ID → RTTM file path (from BUT repo)
  reco2dur  — recording ID → duration in seconds
  ref_rttm  — concatenated RTTM for all recordings (dev/test only)
  uem       — concatenated UEM for all recordings (dev/test only)

Supported conditions:
  SDM1    — Single Distant Microphone ({meeting}.Array1-01.wav, channel 1).
  IHM-MIX — Individual Headset Microphone mix ({meeting}.Mix-Headset.wav).
  MDM8    — All 8 Array1 channels merged via a sox -M pipe.

For meeting ES2010d the Array1-01 / Mix-Headset files are stereo; channel 1 is
extracted with sox remix before writing to a local wav/ subdirectory.

Usage:
  python3 prepare_ami.py <BUT_repo_dir> <AMI_dir> <output_dir> [--cond SDM1|IHM-MIX|MDM8]
"""

import os
import soundfile as sf
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Prepare AMI dataset (unsegmented)')
parser.add_argument('BUT_repo_dir', type=str, help='Path to https://github.com/BUTSpeechFIT/AMI-diarization-setup')
parser.add_argument('AMI_dir', type=str, help='AMI dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='sdm', help='Condition of audio')
args = parser.parse_args()

def main():
    """Generate recording-level Kaldi data directories for all AMI splits.

    Iterates over RTTM files from the BUT repo to determine the meeting list for
    each split. For each meeting:
      - Resolves the audio path based on the requested condition.
      - Handles ES2010d's anomalous stereo file by down-mixing to mono with sox.
      - For MDM8 builds a multi-input sox pipe string covering all 8 channels.
      - Writes wav.scp, utt2spk, rttm.scp, and reco2dur.

    For dev and test splits, additionally concatenates all per-meeting RTTM files
    into ref_rttm and all UEM files into uem (used for scoring with md-eval).
    """
    for split in ["train", "dev", "test"]:
        output_dir = "{}/{}".format(args.output_dir, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(output_dir), 'w')
        rttm_scp_file = open("{}/rttm.scp".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')

        rttm_dir = "{}/only_words/rttms/{}".format(args.BUT_repo_dir, split)
        rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith('.rttm')]
        rttm_files.sort()
        for rttm_file in rttm_files:
            meeting_name = rttm_file.rstrip('.rttm')
            if args.cond == 'SDM1':
                audio_path = "{}/{}/audio/{}.Array1-01.wav".format(args.AMI_dir, meeting_name, meeting_name)
                if not os.path.exists(audio_path):
                    print("{} doesn't exist, skipping it".format(audio_path))
                    continue
                if (sf.info(audio_path)).channels != 1:
                    assert meeting_name == "ES2010d"
                    if not os.path.exists(output_dir + '/wav'):
                        os.makedirs(output_dir + '/wav')
                    output_audio_path = "{}/wav/{}".format(output_dir, audio_path.split('/')[-1])
                    cmd = "sox -t wav {} -t wav {} remix 1".format(audio_path, output_audio_path)
                    print(cmd)
                    status, output = subprocess.getstatusoutput(cmd)
                    assert status == 0
                    audio_path = output_audio_path
                duration = sf.info(audio_path).duration 
            elif args.cond == 'IHM-MIX':
                audio_path = "{}/{}/audio/{}.Mix-Headset.wav".format(args.AMI_dir, meeting_name, meeting_name)
                if not os.path.exists(audio_path):
                    print("{} doesn't exist, skipping it".format(audio_path))
                    continue
                if (sf.info(audio_path)).channels != 1:
                    assert meeting_name == "ES2010d"
                    if not os.path.exists(output_dir + '/wav'):
                        os.makedirs(output_dir + '/wav')
                    output_audio_path = "{}/wav/{}".format(output_dir, audio_path.split('/')[-1])
                    cmd = "sox -t wav {} -t wav {} remix 1".format(audio_path, output_audio_path)
                    print(cmd)
                    status, output = subprocess.getstatusoutput(cmd)
                    assert status == 0
                    audio_path = output_audio_path
                duration = sf.info(audio_path).duration 
            elif args.cond == 'MDM8':
                audio_path = "sox -M"
                missing = False
                for ch in range(1, 9):
                    audio_path_ch = "{}/{}/audio/{}.Array1-0{}.wav".format(args.AMI_dir, meeting_name, meeting_name, ch)
                    if not os.path.exists(audio_path_ch):
                        print("{} doesn't exist, skipping it".format(audio_path_ch))
                        missing = True
                        break
                    if (sf.info(audio_path_ch)).channels != 1:
                        assert meeting_name == "ES2010d"
                        if not os.path.exists(output_dir + '/wav'):
                            os.makedirs(output_dir + '/wav')
                        output_audio_path_ch = "{}/wav/{}".format(output_dir, audio_path_ch.split('/')[-1])
                        cmd = "sox -t wav {} -t wav {} remix 1".format(audio_path_ch, output_audio_path_ch)
                        print(cmd)
                        status, output = subprocess.getstatusoutput(cmd)
                        assert status == 0
                        audio_path_ch = output_audio_path_ch
                    audio_path += " " + audio_path_ch
                if missing:
                    continue
                else:
                    audio_path += " -t wav - |"
                duration = sf.info(audio_path_ch).duration 
            else:
                raise ValueError("Condition not defined.")
            wav_scp_file.write("{} {}\n".format(meeting_name, audio_path))
            utt2spk_file.write("{} {}\n".format(meeting_name, meeting_name))
            rttm_scp_file.write("{} {}/{}\n".format(meeting_name, rttm_dir, rttm_file))
            reco2dur_file.write("{} {}\n".format(meeting_name, duration))

        wav_scp_file.close()
        utt2spk_file.close()
        rttm_scp_file.close()
        reco2dur_file.close()
        
        if split == 'dev' or split == 'test':
            cmd = "cat {}/only_words/rttms/{}/*.rttm > {}/ref_rttm".format(args.BUT_repo_dir, split, output_dir)
            status, output = subprocess.getstatusoutput(cmd)
            assert status == 0
            cmd = "cat {}/uems/{}/*.uem > {}/uem".format(args.BUT_repo_dir, split, output_dir)
            status, output = subprocess.getstatusoutput(cmd)
            assert status == 0
    return 0

if __name__ == '__main__':
    main()

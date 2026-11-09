"""
prepare_alimeeting.py — Prepare the AliMeeting dataset for downstream ASR/diarization tasks.

Reads the raw AliMeeting far-field corpus (multi-channel WAV files and TextGrid
transcription/diarization files) and writes Kaldi-style data directories for the
Eval, Test, and Train splits.

Each output split directory (under {output_dir}/{cond}/{split}/) contains:
  wav.scp   — recording ID → wav path (SDM1: extracted mono channel; MDM8: original)
  utt2spk   — utterance ID → speaker ID
  segments  — utterance ID, recording ID, start time, end time
  text      — utterance ID → M2MeT-normalized transcript
  reco2dur  — recording ID → duration in seconds
  uem       — full-recording UEM spans
  rttm.scp  — recording ID → per-recording RTTM file path

TextGrid tiers encode one speaker per tier; tier names end in _SPK{N}.
Speaker IDs are extracted by stripping the leading "SPK" prefix.

Usage:
  python3 prepare_alimeeting.py <AliMeeting_dir> <output_dir> [--cond SDM1|MDM8]
"""

import os
import soundfile as sf
import argparse
import textgrid
import subprocess

parser = argparse.ArgumentParser(description='Prepare AliMeeting dataset (unsegmented)')
parser.add_argument('AliMeeting_dir', type=str, help='AliMeeting dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
args = parser.parse_args()

def normalize_text_alimeeting(text: str, normalize: str = "m2met") -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    if normalize == "none":
        return text
    elif normalize == "m2met":
        import re

    text = text.replace("<sil>", "")
    text = text.replace("<%>", "")
    text = text.replace("<->", "")
    text = text.replace("<$>", "")
    text = text.replace("<#>", "")
    text = text.replace("<_>", "")
    text = text.replace("<space>", "")
    text = text.replace("`", "")
    text = text.replace("&", "")
    text = text.replace(",", "")
    if re.search("[a-zA-Z]", text):
        text = text.upper()
    text = text.replace("Ａ", "A")
    text = text.replace("ａ", "A")
    text = text.replace("ｂ", "B")
    text = text.replace("ｃ", "C")
    text = text.replace("ｋ", "K")
    text = text.replace("ｔ", "T")
    text = text.replace("，", "")
    text = text.replace("丶", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    return text

def main():
    """Generate Kaldi-style data directories for all AliMeeting splits.

    For each split (Eval, Test, Train):
      - Reads far-field WAV files and corresponding TextGrid annotations.
      - For SDM1 extracts channel 1 via sox remix; for MDM8 uses the original file.
      - Iterates over TextGrid tiers (one per speaker) to produce per-interval
        segments, writing RTTM entries as it goes.
      - Writes wav.scp, utt2spk, segments, text (M2MeT normalized), reco2dur,
        uem, and rttm.scp.

    Utterance IDs follow the pattern: {meet_name}_{speaker}_{start_cs}_{end_cs}
    where start_cs / end_cs are integer centiseconds (7-digit zero-padded).
    """
    for split in ["Eval", "Test", "Train"]:
        split_dir="{}/{}_Ali/{}_Ali_far".format(args.AliMeeting_dir, split, split)
        assert os.path.exists("{}/textgrid_dir".format(split_dir)) and os.path.exists("{}/audio_dir".format(split_dir))

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
        rttm_dir = "{}/rttm".format(output_dir)
        if not os.path.exists(rttm_dir):
            os.makedirs(rttm_dir)
        wav_dir = "{}/wav".format(output_dir)
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')
        text_file = open("{}/text".format(output_dir), 'w')
        uem_file = open("{}/uem".format(output_dir), 'w')
        rttm_scp_file = open("{}/rttm.scp".format(output_dir), 'w')

        audio_files = [f for f in os.listdir("{}/audio_dir".format(split_dir)) if f.endswith('.wav')]
        textgrid_files = [f for f in os.listdir("{}/textgrid_dir".format(split_dir)) if f.endswith('.TextGrid')]
        print("{} meetings in split {}".format(len(audio_files), split))
        audio_files.sort()

        for audio_file in audio_files:
            meet_name = '_'.join(audio_file.strip('.wav').split('_')[:-1])
            audio_file = "{}/audio_dir/{}".format(split_dir, audio_file)
            textgrid_file = meet_name + '.TextGrid'
            textgrid_file = "{}/textgrid_dir/{}".format(split_dir, textgrid_file)
            assert os.path.exists(audio_file) and os.path.exists(textgrid_file)
            tg = textgrid.TextGrid.fromFile(textgrid_file)
            start_t, end_t = tg.minTime, tg.maxTime
            assert start_t == 0
            duration = sf.info(audio_file).duration
            reco2dur_file.write("{} {}\n".format(meet_name, duration))
            uem_file.write("{} 1 {} {}\n".format(meet_name, 0, duration))
            if args.cond == "MDM8":
                wav_scp_file.write("{} {}\n".format(meet_name, audio_file))
            elif args.cond == "SDM1":
                cmd = "sox {} {}/{} remix 1".format(audio_file, wav_dir, audio_file.split('/')[-1])
                status, output = subprocess.getstatusoutput(cmd)
                wav_scp_file.write("{} {}/{}\n".format(meet_name, wav_dir, audio_file.split('/')[-1]))
            else:
                raise ValueError("Condition not defined.")

            rttm_filename = "{}/{}.rttm".format(rttm_dir, meet_name)
            rttm_file = open(rttm_filename, 'w')

            for tier in tg.tiers:
                tier_name_split = tier.name.split('_')
                spkname = tier_name_split[-1]
                assert spkname.startswith('SPK')
                spkname = spkname.lstrip('SPK')

                for i, interval in enumerate(tier.intervals):
                    start_t, end_t = interval.minTime, interval.maxTime
                    assert start_t < end_t
                    text = interval.mark
                    if start_t > duration or end_t > duration:
                        print("Skipping incomplete {}, total duration {:.2f}".format(interval, duration))
                        continue
                    start_t, end_t = round(start_t, 2), round(end_t, 2)
                    segment_name = "{}_{}_{:07d}_{:07d}".format(meet_name, spkname, int(100.0 * start_t), int(100.0 * end_t))
                    segments_file.write("{} {} {:.2f} {:.2f}\n".format(segment_name, meet_name, start_t, end_t))
                    utt2spk_file.write("{} {}\n".format(segment_name, spkname))
                    text = normalize_text_alimeeting(text, normalize="m2met")
                    text_file.write("{} {}\n".format(segment_name, text))
                    rttm_file.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(meet_name, start_t, end_t - start_t, spkname))

            rttm_file.close()
            rttm_scp_file.write("{} {}\n".format(meet_name, rttm_filename))

        wav_scp_file.close()
        utt2spk_file.close()
        reco2dur_file.close()
        segments_file.close()
        text_file.close() 
        uem_file.close()
        rttm_scp_file.close()
    return 0

if __name__ == '__main__':
    main()

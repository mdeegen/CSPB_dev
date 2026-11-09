"""
prepare_mmcsg.py — Prepare the MMCSG dataset for downstream ASR/diarization tasks.

Reads the raw MMCSG corpus (audio, word-level TSV transcriptions, and RTTM diarization
files) and writes Kaldi-style data directories for the dev, eval, and train splits.

Each output split directory contains:
  wav.scp   — recording ID → wav path
  utt2spk   — utterance ID → speaker ID
  segments  — utterance ID, recording ID, start time, end time
  text      — utterance ID → transcript (lowercased)
  reco2dur  — recording ID → duration in seconds
  uem       — full-recording UEM spans
  rttm.scp  — recording ID → RTTM file path (points into the source corpus)

Usage:
  python3 prepare_mmcsg.py <MMCSG_dir> <output_dir> [--cond SDM1|MDM] [--merge_dis FLOAT]
"""

import os
import soundfile as sf
import argparse
import subprocess
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Prepare MMCSG dataset (unsegmented)')
parser.add_argument('MMCSG_dir', type=str, help='MMCSG dataset directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('--cond', type=str, default='SDM1', help='Condition of audio')
parser.add_argument('--merge_dis', type=float, default=0.0, help='Merge distance')
args = parser.parse_args()

def merge_seg(seg_list, merge_dis):
    """Greedily merge consecutive segments from a single speaker.

    Segments are processed in chronological order. A new segment is started
    whenever the gap between the current segment end and the next segment start
    exceeds merge_dis seconds; otherwise the two segments are joined and their
    transcripts concatenated.

    Args:
        seg_list:  List of [start, end, text, speaker] entries, sorted by start
                   time, all belonging to the same speaker.
        merge_dis: Maximum gap in seconds between two segments that will still
                   be merged together. Use 0.0 to only merge overlapping or
                   touching segments.

    Returns:
        List of merged [start, end, text, speaker] entries.
    """
    seg_list_merged = []
    start_t, end_t, text_list = None, None, None
    for i, v in enumerate(seg_list):
        if start_t is None:
            start_t, end_t, text_list = v[0], v[1], [v[2]]
        else:
            if v[0] > end_t + merge_dis:
                seg_list_merged.append([start_t, end_t, ' '.join(text_list), v[3]])
                start_t, end_t, text_list = v[0], v[1], [v[2]]
            else:
                end_t = v[1]
                text_list.append(v[2])
    seg_list_merged.append([start_t, end_t, ' '.join(text_list), v[3]])
    return seg_list_merged

def merge_segments(seg_list, merge_dis):
    """Merge consecutive same-speaker segments across all speakers.

    Splits seg_list by speaker, calls merge_seg on each speaker's segments
    independently, then re-sorts the combined result by start time.

    Args:
        seg_list:  List of [start, end, text, speaker] entries (any order).
        merge_dis: Forwarded to merge_seg; see its docstring.

    Returns:
        Globally time-sorted list of merged [start, end, text, speaker] entries.
    """
    spk_list = [seg[3] for seg in seg_list]
    spk_list = list(set(spk_list))
    spk_list.sort()
    seg_list_merged = []
    for spk in spk_list:
        seg_list_spk = [seg for seg in seg_list if seg[3] == spk]
        seg_list_spk = sorted(seg_list_spk, key=lambda x: x[0])
        seg_list_merged += merge_seg(seg_list_spk, merge_dis)
    seg_list_merged = sorted(seg_list_merged, key=lambda x: x[0])
    return seg_list_merged

def get_segments(fname):
    """Read a word-level TSV transcription file and return merged utterance segments.

    Each line of the TSV has four whitespace-separated fields:
        start_time  end_time  word  speaker_id

    Word-level entries are aggregated into utterance-level segments by
    merge_segments using the global --merge_dis threshold.

    Args:
        fname: Path to the TSV transcription file.

    Returns:
        List of [start, end, text, speaker] entries (utterance level).
    """
    with open(fname, 'r') as fh:
        content = fh.readlines()
    seg_list = []
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        assert len(line_split) == 4
        start_t, end_t, word, spk = line_split
        start_t = round(float(start_t), 2)
        end_t = round(float(end_t), 2)
        seg_list.append([start_t, end_t, word, spk])
    seg_list_merged = merge_segments(seg_list, args.merge_dis)
    return seg_list_merged

def main():
    """Generate Kaldi-style data directories for all MMCSG splits.

    For each split (dev, eval, train):
      - Reads word-level TSV transcriptions and merges them into utterance segments.
      - Resamples audio to 16 kHz; for SDM1 extracts channel 1 via sox remix.
      - Writes wav.scp, utt2spk, segments, text, reco2dur, uem, and rttm.scp.
        rttm.scp points directly into the source corpus RTTM files (no copy).

    Utterance IDs follow the pattern: {meet_name}_{speaker}_{start_cs}_{end_cs}
    where start_cs / end_cs are integer centiseconds (7-digit zero-padded).
    """
    for split in ["dev", "eval", "train"]:
        audio_dir = "{}/audio/{}".format(args.MMCSG_dir, split)
        text_dir = "{}/transcriptions/{}".format(args.MMCSG_dir, split)
        rttm_dir = "{}/rttm/{}".format(args.MMCSG_dir, split)

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
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

        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        print("{} meetings in split {}".format(len(audio_files), split))
        audio_files.sort()

        for audio_file in tqdm(audio_files):
            meet_name = audio_file.strip('.wav')
            audio_file = "{}/audio/{}/{}".format(args.MMCSG_dir, split, audio_file)
            tsv_file = "{}/transcriptions/{}/{}.tsv".format(args.MMCSG_dir, split, meet_name)
            rttm_file = "{}/rttm/{}/{}.rttm".format(args.MMCSG_dir, split, meet_name)
            assert os.path.exists(audio_file) and os.path.exists(tsv_file) and os.path.exists(rttm_file)
            segments = get_segments(tsv_file)

            duration = sf.info(audio_file).duration
            channels = sf.info(audio_file).channels
            reco2dur_file.write("{} {}\n".format(meet_name, duration))
            uem_file.write("{} 1 {} {}\n".format(meet_name, 0, duration))
            if args.cond == "MDM":
                cmd = "sox {} -r 16000 {}/{}".format(audio_file, wav_dir, audio_file.split('/')[-1])
                status, output = subprocess.getstatusoutput(cmd)
                wav_scp_file.write("{} {}/{}\n".format(meet_name, wav_dir, audio_file.split('/')[-1]))
            elif args.cond == "SDM1":
                cmd = "sox {} -r 16000 {}/{} remix 1".format(audio_file, wav_dir, audio_file.split('/')[-1])
                status, output = subprocess.getstatusoutput(cmd)
                wav_scp_file.write("{} {}/{}\n".format(meet_name, wav_dir, audio_file.split('/')[-1]))
            else:
                raise ValueError("Condition not defined.")

            for i, seg in enumerate(segments):
                start_t, end_t = round(seg[0], 2), round(seg[1], 2)
                text, spkname = seg[2].lower(), seg[3]
                segment_name = "{}_{}_{:07d}_{:07d}".format(meet_name, spkname, int(100.0 * start_t), int(100.0 * end_t))
                segments_file.write("{} {} {:.2f} {:.2f}\n".format(segment_name, meet_name, start_t, end_t))
                utt2spk_file.write("{} {}\n".format(segment_name, spkname))
                text_file.write("{} {}\n".format(segment_name, text))

            rttm_scp_file.write("{} {}\n".format(meet_name, rttm_file))

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

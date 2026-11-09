"""
prepare_dipco.py — Prepare the DiPCo (Dinner Party Corpus) dataset for downstream
ASR/diarization tasks.

Reads the raw DiPCo corpus (multi-channel WAV files and per-session JSON transcription
files) and writes Kaldi-style data directories for the dev and eval splits.

DiPCo has 5 microphone array units (U01–U05) per session; each unit has 7 channels
(CH1–CH7). Recording IDs are formed as {session}_{mic} (e.g. S01_U01).

Each output split directory (under {output_dir}/{cond}/{split}/) contains:
  wav.scp   — recording ID → wav path (MDM: merged 7-channel file; SDM1: CH7 only)
  utt2spk   — utterance ID → speaker ID
  segments  — utterance ID, recording ID, start time, end time
  text      — utterance ID → transcript (lowercased, noise tags removed)
  reco2dur  — recording ID → duration in seconds
  uem       — full-recording UEM spans
  rttm.scp  — recording ID → per-recording RTTM file path

Utterance IDs follow the pattern: {meet_name}_{speaker}_{start_cs}_{end_cs}
where start_cs / end_cs are integer centiseconds (7-digit zero-padded).

Usage:
  python3 prepare_dipco.py <DiPCo_dir> <output_dir> [--cond SDM1|MDM] [--merge_dis FLOAT]
"""

import os
import soundfile as sf
import argparse
import subprocess
from tqdm import tqdm
import json
import numpy as np

parser = argparse.ArgumentParser(description='Prepare DiPCo dataset (unsegmented)')
parser.add_argument('DiPCo_dir', type=str, help='DiPCo dataset directory')
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

def load_label(json_file):
    """Load a DiPCo JSON annotation file and return merged utterance segments.

    Each JSON entry contains per-microphone start/end timestamps (U01–U05 and
    close-talk). The timestamps are asserted to be identical across all mics.
    Non-lexical noise tokens ([unintelligible], [laugh], [noise]) are filtered
    from the transcript before segment merging.

    Args:
        json_file: Path to the session-level JSON annotation file.

    Returns:
        List of merged [start, end, text, speaker] entries (utterance level).
    """
    seg_list = []
    with open(json_file, "r") as f:
        label = json.load(f)
    for line in label:
        start_t_list = [convert_time(line['start_time'][mic]) for mic in ["U01", "U02", "U03", "U04", "U05", "close-talk"]]
        end_t_list = [convert_time(line['end_time'][mic]) for mic in ["U01", "U02", "U03", "U04", "U05", "close-talk"]]
        # Sanity-check: timestamps must agree across all microphones
        assert np.max(start_t_list) == np.min(start_t_list) and np.max(end_t_list) == np.min(end_t_list)
        start_t, end_t = np.max(start_t_list), np.max(end_t_list)
        word_list = line['words'].split()
        word_list = [w for w in word_list if w not in ['[unintelligible]', '[laugh]', '[noise]']]
        seg_list.append([start_t, end_t, ' '.join(word_list), line['speaker_id']])
    seg_list_merged = merge_segments(seg_list, args.merge_dis)
    return seg_list_merged

def convert_time(time_str):
    """Convert a HH:MM:SS.fff timestamp string to total seconds (float).

    Args:
        time_str: Timestamp in "HH:MM:SS.fff" format.

    Returns:
        Equivalent duration in seconds as a float.
    """
    h, m, s = time_str.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return seconds

def main():
    """Generate Kaldi-style data directories for all DiPCo splits.

    For each split (dev, eval) and each of the 5 microphone array units (U01–U05):
      - Loads utterance segments from the session-level JSON annotation.
      - For MDM merges all 7 channels with sox --combine merge into a local wav file.
      - For SDM1 uses the CH7 file directly without copying.
      - Writes wav.scp, utt2spk, segments, text, reco2dur, uem, and rttm.scp.
        RTTM files are written per recording to {output_dir}/rttm/.

    Segment timestamps are shared across all mic units within a session; each unit
    gets its own copy of the segments/text/rttm files pointing at its own audio.
    """
    for split in ["dev", "eval"]:
        audio_dir = "{}/audio/{}".format(args.DiPCo_dir, split)
        text_dir = "{}/transcriptions/{}".format(args.DiPCo_dir, split)

        output_dir = "{}/{}/{}".format(args.output_dir, args.cond, split)
        wav_dir = "{}/wav".format(output_dir)
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        rttm_dir = "{}/rttm".format(output_dir)
        if not os.path.exists(rttm_dir):
            os.makedirs(rttm_dir)

        wav_scp_file = open("{}/wav.scp".format(output_dir), 'w')
        utt2spk_file = open("{}/utt2spk".format(output_dir), 'w')
        reco2dur_file = open("{}/reco2dur".format(output_dir), 'w')
        segments_file = open("{}/segments".format(output_dir), 'w')
        text_file = open("{}/text".format(output_dir), 'w')
        uem_file = open("{}/uem".format(output_dir), 'w')
        rttm_scp_file = open("{}/rttm.scp".format(output_dir), 'w')

        sessions = list(set([f.split('_')[0] for f in os.listdir(audio_dir) if f.endswith('.wav') and f.startswith('S')]))
        print("{} meetings in split {}".format(len(sessions), split))
        sessions.sort()

        for sess_name in tqdm(sessions):
            duration_list = []
            json_file = "{}/transcriptions/{}/{}.json".format(args.DiPCo_dir, split, sess_name)
            assert os.path.exists(json_file)
            segments = load_label(json_file)
            for mic in ["U01", "U02", "U03", "U04", "U05"]: 
                meet_name = "{}_{}".format(sess_name, mic)
                audio_file = "{}/audio/{}/{}.CH7.wav".format(args.DiPCo_dir, split, meet_name)
                assert os.path.exists(audio_file)

                duration = sf.info(audio_file).duration
                channels = sf.info(audio_file).channels
                reco2dur_file.write("{} {}\n".format(meet_name, duration))
                uem_file.write("{} 1 {} {}\n".format(meet_name, 0, duration))

                if args.cond == "MDM":
                    cmd = "sox --combine merge "
                    for chan in ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7"]:
                        audio_file_chan = audio_file.replace("CH7", chan)
                        assert os.path.exists(audio_file_chan)
                        cmd += " {}".format(audio_file_chan)
                    output_audio_file = "{}/{}.wav".format(wav_dir, meet_name)
                    cmd += " {}".format(output_audio_file)
                    print(cmd)
                    status, output = subprocess.getstatusoutput(cmd)
                    assert status == 0
                    wav_scp_file.write("{} {}\n".format(meet_name, output_audio_file))
                elif args.cond == "SDM1":
                    wav_scp_file.write("{} {}\n".format(meet_name, audio_file))
                else:
                    raise ValueError("Condition not defined.")

                rttm_filename = "{}/{}.rttm".format(rttm_dir, meet_name)
                rttm_file = open(rttm_filename, 'w')
                for i, seg in enumerate(segments):
                    start_t, end_t = round(seg[0], 2), round(seg[1], 2)
                    text, spkname = seg[2].lower(), seg[3]
                    segment_name = "{}_{}_{:07d}_{:07d}".format(meet_name, spkname, int(100.0 * start_t), int(100.0 * end_t))
                    segments_file.write("{} {} {:.2f} {:.2f}\n".format(segment_name, meet_name, start_t, end_t))
                    utt2spk_file.write("{} {}\n".format(segment_name, spkname))
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

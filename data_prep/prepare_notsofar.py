"""Prepare NotSoFar multi-channel recordings for Kaldi-style ASR data.

Each selected ``mc_*`` device becomes an independent recording.
Its selected channel WAVs are merged into one 16-kHz WAV, while the JSON
segments are reused for that recording.  This follows the multi-array
handling used by DiPCo and avoids silently mixing independent devices.

Examples:
    python3 data_prep/prepare_notsofar.py --output-root OUTPUT \
            --split train=ROOT/train
  python3 data_prep/prepare_notsofar.py ROOT OUTPUT --device mc_plaza_0
  python3 data_prep/prepare_notsofar.py ROOT OUTPUT --channels ch0,ch2
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import soundfile as sf

try:
    from notsofar_text import normalize_transcript
except ImportError:
    from data_prep.notsofar_text import normalize_transcript


MEETING_JSON = "gt_transcription.json"
CHANNEL_PATTERN = re.compile(r"^ch(\d+)\.wav$")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NotSoFar ASR data")
    parser.add_argument(
        "input_root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="NotSoFar root or split root (legacy positional form)",
    )
    parser.add_argument(
        "legacy_output_root",
        type=Path,
        nargs="?",
        help="Output root (legacy positional form)",
    )
    parser.add_argument("--output-root", type=Path, help="Output root")
    parser.add_argument(
        "--input-root",
        dest="input_root_option",
        type=Path,
        help="NotSoFar root when --split is not used",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Explicit split root; repeat for train/dev/test",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        help="Device directory name, e.g. mc_plaza_0; repeat or omit for all",
    )
    parser.add_argument(
        "--channels",
        default="all",
        help="Comma-separated channel files (ch0,ch2) or 'all'",
    )
    parser.add_argument(
        "--condition",
        default="MDM",
        help="Output condition name (default: MDM)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first malformed annotation instead of skipping it",
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = args.legacy_output_root
    if args.input_root_option is not None:
        args.input_root = args.input_root_option
    if args.output_root is None:
        parser.error("provide --output-root (or legacy positional output root)")
    return args


def split_roots(input_root: Path, definitions: list[str]) -> list[tuple[str, Path]]:
    """Resolve explicit split definitions or discover split-like directories."""
    if definitions:
        result = []
        for definition in definitions:
            if "=" not in definition:
                raise ValueError("--split must use NAME=PATH")
            name, path = definition.split("=", 1)
            if not name or not path:
                raise ValueError("--split must use a non-empty NAME=PATH")
            result.append((name, Path(path)))
        return result

    if (input_root / MEETING_JSON).is_file():
        return [(input_root.name, input_root)]
    candidates = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and any(path.rglob(MEETING_JSON))
    ]
    if candidates:
        return [(path.name, path) for path in sorted(candidates)]
    return [(input_root.name, input_root)]


def meeting_dirs(split_root: Path) -> list[Path]:
    return sorted({path.parent for path in split_root.rglob(MEETING_JSON)})


def selected_devices(meeting_dir: Path, requested: list[str]) -> list[Path]:
    devices = sorted(
        path
        for path in meeting_dir.iterdir()
        if path.is_dir() and path.name.startswith("mc_")
    )
    if requested:
        requested_set = set(requested)
        devices = [path for path in devices if path.name in requested_set]
        missing = requested_set - {path.name for path in devices}
        if missing:
            raise FileNotFoundError(
                "Missing device(s) in {}: {}".format(
                    meeting_dir, ", ".join(sorted(missing))
                )
            )
    if not devices:
        raise FileNotFoundError(
            "No multi-channel mc_* device directories in {}".format(meeting_dir)
        )
    return devices


def channel_number(path: Path) -> int:
    match = CHANNEL_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Invalid channel filename: {path.name}")
    return int(match.group(1))


def selected_channels(device_dir: Path, channel_spec: str) -> list[Path]:
    channels = sorted(
        (
            path
            for path in device_dir.glob("ch*.wav")
            if CHANNEL_PATTERN.match(path.name)
        ),
        key=channel_number,
    )
    if channel_spec != "all":
        names = [name.strip() for name in channel_spec.split(",") if name.strip()]
        channels = [
            device_dir / (name if name.endswith(".wav") else name + ".wav")
            for name in names
        ]
        missing = [path.name for path in channels if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing channel(s) in {}: {}".format(device_dir, ", ".join(missing))
            )
    if not channels:
        raise FileNotFoundError("No ch*.wav files in {}".format(device_dir))
    if len(channels) < 2:
        raise ValueError(
            "Device {} is not multi-channel: found {} channel".format(
                device_dir, len(channels)
            )
        )
    return channels


def load_segments(json_path: Path, strict: bool):
    with json_path.open(encoding="utf-8") as file:
        annotations = json.load(file)
    segments = []
    rejected = []
    stats = {
        "total": len(annotations),
        "valid": 0,
        "annotation_only": 0,
        "empty_source": 0,
        "malformed": 0,
    }
    for line_number, annotation in enumerate(annotations, start=1):
        try:
            start = float(annotation["start_time"])
            end = float(annotation["end_time"])
            speaker = str(annotation["speaker_id"])
            source_text = annotation.get("text")
            if source_text is None or not str(source_text).strip():
                stats["empty_source"] += 1
                reason = "empty source transcript"
                raise ValueError(reason)
            text = normalize_transcript(str(source_text))
            if not text:
                stats["annotation_only"] += 1
                reason = "annotation-only transcript"
                raise ValueError(reason)
            if not 0 <= start < end:
                raise ValueError("invalid time range")
            segments.append((start, end, text, speaker))
            stats["valid"] += 1
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            message = "{}:{}: {}".format(json_path, line_number, error)
            if strict:
                raise ValueError(message) from error
            rejected.append(
                {
                    "file": str(json_path),
                    "line": line_number,
                    "reason": str(error),
                    "annotation": annotation,
                }
            )
            print("Skipping {}".format(message))
            if "annotation-only" not in str(error) and "empty source" not in str(error):
                stats["malformed"] += 1
    return segments, stats, rejected


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def merge_channels(channel_files: list[Path], output_file: Path) -> tuple[float, int]:
    """Merge channels with SoX and return duration and channel count."""
    infos = [sf.info(path) for path in channel_files]
    if len({info.samplerate for info in infos}) != 1:
        raise ValueError("Channel sample rates differ: {}".format(channel_files))
    if max(info.frames for info in infos) - min(info.frames for info in infos) > 1:
        raise ValueError("Channel durations differ: {}".format(channel_files))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "sox",
        "--combine",
        "merge",
        *[str(path) for path in channel_files],
        "-r",
        "16000",
        str(output_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "SoX failed for {}: {}".format(
                channel_files[0].parent, result.stderr.strip()
            )
        )
    info = sf.info(output_file)
    return info.duration, info.channels


def write_split(split_name, split_root, output_root, args):
    output_dir = output_root / args.condition / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        name: (output_dir / name).open("w", encoding="utf-8")
        for name in ("wav.scp", "segments", "text", "utt2spk", "reco2dur")
    }
    rejected_file = (output_dir / "rejected.jsonl").open("w", encoding="utf-8")
    totals = {
        "total": 0,
        "valid": 0,
        "annotation_only": 0,
        "empty_source": 0,
        "malformed": 0,
        "out_of_audio": 0,
    }
    seen_ids = set()
    try:
        for meeting_dir in meeting_dirs(split_root):
            json_path = meeting_dir / MEETING_JSON
            segments, stats, rejected = load_segments(json_path, args.strict)
            for key, value in stats.items():
                totals[key] += value
            for item in rejected:
                rejected_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            for device_dir in selected_devices(meeting_dir, args.device):
                channels = selected_channels(device_dir, args.channels)
                recording_id = safe_id(
                    "{}_{}".format(meeting_dir.name, device_dir.name)
                )
                output_wav = output_dir / "wav" / (recording_id + ".wav")
                duration, channel_count = merge_channels(channels, output_wav)
                files["wav.scp"].write("{} {}\n".format(recording_id, output_wav))
                files["reco2dur"].write("{} {:.6f}\n".format(recording_id, duration))
                for index, (start, end, text, speaker) in enumerate(segments):
                    if end > duration + 0.01:
                        message = "{} segment ends at {:.3f}, audio is {:.3f}".format(
                            recording_id, end, duration
                        )
                        if args.strict:
                            raise ValueError(message)
                        print("Skipping {}".format(message))
                        totals["out_of_audio"] += 1
                        rejected_file.write(
                            json.dumps(
                                {
                                    "file": str(json_path),
                                    "reason": "out of audio",
                                    "recording": recording_id,
                                    "start": start,
                                    "end": end,
                                }
                            )
                            + "\n"
                        )
                        continue
                    segment_id = "{}_{}_{}_{}".format(
                        recording_id,
                        safe_id(speaker),
                        int(round(start * 100)),
                        int(round(end * 100)),
                    )
                    if segment_id in seen_ids:
                        segment_id = "{}_{}".format(segment_id, index)
                    seen_ids.add(segment_id)
                    files["segments"].write(
                        "{} {} {:.2f} {:.2f}\n".format(
                            segment_id, recording_id, start, end
                        )
                    )
                    files["text"].write("{} {}\n".format(segment_id, text))
                    files["utt2spk"].write("{} {}\n".format(segment_id, speaker))
                print(
                    "{}: {} channels, {:.2f}s".format(
                        recording_id, channel_count, duration
                    )
                )
    finally:
        for file in files.values():
            file.close()
        rejected_file.close()
    print(
        "{}: annotations={} valid={} annotation_only={} empty_source={} "
        "malformed={} out_of_audio={}".format(
            split_name,
            totals["total"],
            totals["valid"],
            totals["annotation_only"],
            totals["empty_source"],
            totals["malformed"],
            totals["out_of_audio"],
        )
    )


def main():
    args = parse_args()
    for split_name, split_root in split_roots(args.input_root, args.split):
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        write_split(split_name, split_root, args.output_root, args)


if __name__ == "__main__":
    main()

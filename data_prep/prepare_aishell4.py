"""Prepare AISHELL-4 FLAC/TextGrid data for Kaldi-style ASR.

The source layout is expected to be::

    ROOT/{train_L,train_M,train_S,test}/{wav,TextGrid}/<recording>.(flac|wav)

The three train source groups are combined into one output ``train`` split.
Audio is converted to 16-kHz WAV while preserving its channel order. TextGrid
intervals provide ASR segments; matching RTTM files are used to resolve
canonical speaker IDs when their time overlap is unambiguous.
"""

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

TEXTGRID_SUFFIX = ".TextGrid"
RTTM_SUFFIX = ".rttm"
AUDIO_SUFFIXES = (".flac", ".wav")
_NON_SPEECH = re.compile(r"^\s*(?:<[^>]+>|\[[^]]+\]|sil|silence|sp)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class RttmSegment:
    start: float
    end: float
    speaker: str


@dataclass
class Interval:
    minTime: float
    maxTime: float
    mark: str


@dataclass
class Tier:
    name: str
    intervals: list[Interval]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AISHELL-4 ASR data")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-split",
        action="append",
        default=[],
        help="Source directory name; repeat (default: train_L train_M train_S test)",
    )
    parser.add_argument("--train-output-name", default="train")
    parser.add_argument("--audio-dir-name", default="wav")
    parser.add_argument("--annotation-dir-name", default="TextGrid")
    parser.add_argument(
        "--rttm-mode", choices=("validate", "copy", "ignore"), default="validate"
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize an AISHELL transcript without deleting Chinese characters."""
    text = re.sub(r"\s+", " ", text.replace("\ufeff", "")).strip()
    if _NON_SPEECH.fullmatch(text):
        return ""
    return text.upper()


def find_audio(audio_dir: Path, stem: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = audio_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def parse_rttm(path: Path) -> list[RttmSegment]:
    segments = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 8 or fields[0] != "SPEAKER":
                raise ValueError(f"Malformed RTTM line {path}:{line_number}")
            start = float(fields[3])
            duration = float(fields[4])
            if start < 0 or duration <= 0:
                raise ValueError(f"Invalid RTTM timing {path}:{line_number}")
            segments.append(RttmSegment(start, start + duration, fields[7]))
    return segments


def speaker_for_interval(start: float, end: float, rttm: list[RttmSegment]):
    overlaps = {}
    for segment in rttm:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap:
            overlaps[segment.speaker] = overlaps.get(segment.speaker, 0.0) + overlap
    if not overlaps:
        return None, "no_rttm_overlap"
    ranked = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, "ambiguous_rttm_overlap"
    return ranked[0][0], "rttm"


def convert_audio(source: Path, target: Path) -> tuple[float, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_info = sf.info(source)
    if source_info.samplerate == 16000:
        audio, _ = sf.read(source, always_2d=True, dtype="float32")
        sf.write(target, audio, 16000, subtype="PCM_16", format="WAV")
    else:
        result = subprocess.run(
            ["sox", str(source), "-r", "16000", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"SoX conversion failed for {source}: {result.stderr.strip()}"
            )
    info = sf.info(target)
    return info.duration, info.channels


def load_textgrid(path: Path):
    tiers = []
    current_tier = None
    current_interval = None
    with path.open(encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("item ["):
                if current_tier is not None:
                    tiers.append(current_tier)
                current_tier = Tier(name="", intervals=[])
                current_interval = None
            elif current_tier is not None and line.startswith("name ="):
                current_tier.name = parse_quoted_value(line)
            elif current_tier is not None and line.startswith("intervals ["):
                current_interval = Interval(0.0, 0.0, "")
                current_tier.intervals.append(current_interval)
            elif current_interval is not None and line.startswith("xmin ="):
                current_interval.minTime = float(line.split("=", 1)[1].strip())
            elif current_interval is not None and line.startswith("xmax ="):
                current_interval.maxTime = float(line.split("=", 1)[1].strip())
            elif current_interval is not None and line.startswith("text ="):
                current_interval.mark = parse_quoted_value(line)
    if current_tier is not None:
        tiers.append(current_tier)
    if not tiers:
        raise ValueError(f"No interval tiers found in {path}")
    return tiers


def parse_quoted_value(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    return value


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff.-]+", "-", value).strip("-") or "unknown"


def write_split(
    source_root: Path,
    source_name: str,
    output_root: Path,
    args,
    output_name: str,
    append: bool,
):
    audio_dir = source_root / args.audio_dir_name
    annotation_dir = source_root / args.annotation_dir_name
    if not audio_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing {audio_dir} or {annotation_dir}")

    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        name: (output_dir / name).open("a" if append else "w", encoding="utf-8")
        for name in ("wav.scp", "segments", "text", "utt2spk", "reco2dur")
    }
    rttm_scp = (
        (output_dir / "rttm.scp").open("a" if append else "w", encoding="utf-8")
        if args.rttm_mode == "copy"
        else None
    )
    rejected = (output_dir / "rejected.jsonl").open(
        "a" if append else "w", encoding="utf-8"
    )
    stats = {
        "recordings": 0,
        "annotations": 0,
        "valid": 0,
        "empty": 0,
        "out_of_audio": 0,
        "fallback_speaker": 0,
        "rttm_mapped": 0,
    }
    try:
        textgrids = sorted(annotation_dir.glob(f"*{TEXTGRID_SUFFIX}"))
        for textgrid_path in textgrids:
            stem = textgrid_path.name[: -len(TEXTGRID_SUFFIX)]
            audio_path = find_audio(audio_dir, stem)
            rttm_path = annotation_dir / f"{stem}{RTTM_SUFFIX}"
            if audio_path is None:
                raise FileNotFoundError(f"Missing audio for {textgrid_path}")
            if args.rttm_mode != "ignore" and not rttm_path.is_file():
                message = {"file": str(textgrid_path), "reason": "missing_rttm"}
                rejected.write(json.dumps(message, ensure_ascii=False) + "\n")
                if args.strict:
                    raise FileNotFoundError(message["file"])
            rttm = parse_rttm(rttm_path) if rttm_path.is_file() else []
            recording_id = f"{source_name}_{stem}"
            output_audio = output_dir / "wav" / f"{safe_id(recording_id)}.wav"
            duration, channels = convert_audio(audio_path, output_audio)
            files["wav.scp"].write(f"{recording_id} {output_audio}\n")
            files["reco2dur"].write(f"{recording_id} {duration:.6f}\n")
            stats["recordings"] += 1
            grid = load_textgrid(textgrid_path)
            for tier_index, tier in enumerate(grid):
                tier_name = tier.name.strip() or f"tier{tier_index}"
                for interval_index, interval in enumerate(tier.intervals):
                    stats["annotations"] += 1
                    start, end = float(interval.minTime), float(interval.maxTime)
                    raw_text = str(interval.mark or "")
                    text = normalize_text(raw_text)
                    base_reject = {
                        "file": str(textgrid_path),
                        "tier": tier_name,
                        "interval": interval_index,
                        "start": start,
                        "end": end,
                        "text": raw_text,
                    }
                    if end <= start or not text:
                        stats["empty"] += 1
                        base_reject["reason"] = "empty_or_zero_length"
                        rejected.write(
                            json.dumps(base_reject, ensure_ascii=False) + "\n"
                        )
                        continue
                    if start < 0 or end > duration + 0.01:
                        stats["out_of_audio"] += 1
                        base_reject["reason"] = "out_of_audio"
                        rejected.write(
                            json.dumps(base_reject, ensure_ascii=False) + "\n"
                        )
                        if args.strict:
                            raise ValueError(base_reject)
                        continue
                    speaker, mapping = speaker_for_interval(start, end, rttm)
                    if speaker is None:
                        speaker = tier_name
                        stats["fallback_speaker"] += 1
                    else:
                        stats["rttm_mapped"] += 1
                    start_cs, end_cs = round(start * 100), round(end * 100)
                    utterance_id = (
                        f"{recording_id}_{safe_id(speaker)}_{start_cs:07d}_{end_cs:07d}"
                    )
                    files["segments"].write(
                        f"{utterance_id} {recording_id} {start:.2f} {end:.2f}\n"
                    )
                    files["text"].write(f"{utterance_id} {text}\n")
                    files["utt2spk"].write(f"{utterance_id} {speaker}\n")
                    stats["valid"] += 1
            if args.rttm_mode == "copy" and rttm_path.is_file():
                rttm_output = output_dir / "rttm" / f"{recording_id}.rttm"
                rttm_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(rttm_path, rttm_output)
                rttm_scp.write(f"{recording_id} {rttm_output}\n")
        print(
            f"{output_name}/{source_name}: {stats}, channels checked={channels if stats['recordings'] else 0}"
        )
    finally:
        for file in files.values():
            file.close()
        if rttm_scp is not None:
            rttm_scp.close()
        rejected.close()


def main():
    args = parse_args()
    source_names = args.source_split or ["train_L", "train_M", "train_S", "test"]
    initialized = set()
    for source_name in source_names:
        source_root = args.input_root / source_name
        output_name = "test" if source_name == "test" else args.train_output_name
        write_split(
            source_root,
            source_name,
            args.output_root,
            args,
            output_name,
            append=output_name in initialized,
        )
        initialized.add(output_name)


if __name__ == "__main__":
    main()

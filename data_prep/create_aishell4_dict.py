"""Create an AISHELL-4 character dictionary from prepared training text."""

import argparse
from collections import Counter
from pathlib import Path

from notsofar_text import transcript_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Create an AISHELL-4 character dictionary"
    )
    parser.add_argument("--text-file", action="append", type=Path, default=[])
    parser.add_argument("--data-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    text_files = list(args.text_file)
    text_files.extend(path / "text" for path in args.data_dir)
    if not text_files:
        parser.error("provide at least one --text-file or --data-dir")
    counts = Counter()
    utterances = 0
    for text_file in text_files:
        with text_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.rstrip("\n\r").split(None, 1)
                if len(fields) != 2:
                    raise ValueError(f"Malformed line in {text_file}:{line_number}")
                counts.update(transcript_tokens(fields[1]))
                utterances += 1
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as file:
        for symbol, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            file.write(f"{symbol} {count}\n")
    print(
        f"Created {args.output_file} from {utterances} utterances ({len(counts)} symbols)"
    )


if __name__ == "__main__":
    main()

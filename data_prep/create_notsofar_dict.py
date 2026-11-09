"""Create a NotSoFar character dictionary from prepared Kaldi text files."""

import argparse
from collections import Counter
from pathlib import Path

try:
    from notsofar_text import transcript_tokens
except ImportError:
    from data_prep.notsofar_text import transcript_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Create a NotSoFar character dictionary"
    )
    parser.add_argument("text_files", type=Path, nargs="*")
    parser.add_argument("positional_output", type=Path, nargs="?")
    parser.add_argument(
        "--text-file",
        type=Path,
        action="append",
        default=[],
        help="Kaldi text file; repeat for multiple training files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=[],
        help="Prepared Kaldi directory containing text",
    )
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    text_files = list(args.text_files) + list(args.text_file)
    text_files.extend(data_dir / "text" for data_dir in args.data_dir)
    output_file = args.output_file or args.positional_output
    if (
        output_file is None
        and not args.data_dir
        and not args.text_file
        and len(text_files) >= 2
    ):
        output_file = text_files.pop()
    if not text_files or output_file is None:
        parser.error("provide text files and an output file")
    counts = Counter()
    utterances = 0
    for text_file in text_files:
        with text_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.rstrip("\n\r").split(None, 1)
                if len(fields) != 2:
                    raise ValueError(
                        "Malformed line in {}:{}".format(text_file, line_number)
                    )
                counts.update(transcript_tokens(fields[1]))
                utterances += 1
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for symbol, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            file.write("{} {}\n".format(symbol, count))
    print(
        "Created {} from {} utterances ({} symbols)".format(
            output_file, utterances, len(counts)
        )
    )


if __name__ == "__main__":
    main()

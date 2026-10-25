"""Regenerate API pages and build documentation using the installed package."""

import argparse
from pathlib import Path
import subprocess
import sys

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS.parent / "src" / "pysolver_view"
BUILD = DOCS / "_build"


def build_docs(*, preview=False, output_dir=BUILD):
    """Build from any working directory; source paths do not imply import paths."""
    build_path = Path(output_dir).resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx.ext.apidoc",
            "-e",
            "-M",
            "-f",
            "-o",
            str(DOCS / "source" / "api"),
            str(SOURCE),
        ],
        check=True,
    )
    if preview:
        command = [
            sys.executable,
            "-m",
            "sphinx_autobuild",
            str(DOCS),
            str(build_path),
            "--open-browser",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-a",
            "-b",
            "html",
            str(DOCS),
            str(build_path),
        ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-autobuild", action="store_true", help="Build once without a live preview."
    )
    parser.add_argument("--output-dir", type=Path, default=BUILD)
    args = parser.parse_args()
    build_docs(preview=not args.no_autobuild, output_dir=args.output_dir)

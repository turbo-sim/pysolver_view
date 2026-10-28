#!/usr/bin/env python3
"""Master test runner for local development and the existing CI jobs.

Edit TEST_SUITES, ENABLED_SUITES and PYTEST_OPTIONS below to control the checks.
Additional command-line arguments are passed to pytest.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_SUITES = {
    "package": ["tests"],
    "studies": [
        "tests/roots/test_convergence_benchmark.py",
        "tests/roots/test_timing_benchmark.py",
        "tests/optimization/test_slsqp_controls_study.py",
        "tests/optimization/test_pounce_studies.py",
    ],
}
ENABLED_SUITES = ["package"]
# Hessian validation remains outside the agreed migration scope.
PYTEST_OPTIONS = ["-q", "-k", "not hessian"]
CACHE_DIR = REPO_ROOT / "build" / "pytest_cache"


def main(argv=None):
    """Run the configured suites from the repository and return pytest's status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=TEST_SUITES,
        action="append",
        help="Run only this suite; repeat to select several. Default: all enabled suites.",
    )
    options, pytest_args = parser.parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    suites = options.suite if options.suite is not None else ENABLED_SUITES
    paths = [str(REPO_ROOT / path) for suite in suites for path in TEST_SUITES[suite]]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            *PYTEST_OPTIONS,
            "-o",
            f"cache_dir={CACHE_DIR}",
            *pytest_args,
        ],
        cwd=REPO_ROOT,
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())

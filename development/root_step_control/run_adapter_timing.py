"""Measure public root-interface overhead against direct calls to the same core.

This is an interface experiment, separate from comparisons with SciPy/MINPACK.
Run directly to set one BLAS thread before importing numerical libraries.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

if __name__ == "__main__":
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from development.root_step_control import run_timing_benchmark as timing


def summarize(rows):
    """Pair clean medians and retain additive instrumented time components."""
    summary = []
    for problem, solver in sorted({(row["problem"], row["solver"]) for row in rows}):
        entry = dict(problem=problem, solver=solver)
        group = [
            row for row in rows if (row["problem"], row["solver"]) == (problem, solver)
        ]
        entry["all_root_success"] = all(row["all_root_success"] for row in group)
        for interface in ("core", "public"):
            selected = [row for row in group if row["interface"] == interface]
            clean = [
                row["total_ns"] / 1e6 for row in selected if not row["instrumented"]
            ]
            entry[f"{interface}_median_ms"] = float(np.median(clean))
            entry[f"{interface}_q25_ms"], entry[f"{interface}_q75_ms"] = np.percentile(
                clean, [25, 75]
            )
            measured = sorted(
                (row for row in selected if row["instrumented"]),
                key=lambda row: row["total_ns"],
            )
            representative = measured[len(measured) // 2]
            for field in ("total", "residual", "jacobian", *timing.KERNELS):
                entry[f"{interface}_{field}_ms"] = representative[f"{field}_ns"] / 1e6
            entry[f"{interface}_other_ms"] = (
                representative["solver_overhead_ns"]
                - sum(representative[f"{kernel}_ns"] for kernel in timing.KERNELS)
            ) / 1e6
            for field in ("residual_calls", "jacobian_calls", "accepted_iterations"):
                values = {row[field] for row in measured}
                if len(values) != 1:
                    raise AssertionError(
                        f"Unstable work counts: {problem}/{solver}/{interface}"
                    )
                entry[f"{interface}_{field}"] = values.pop()
        for field in ("residual_calls", "jacobian_calls", "accepted_iterations"):
            if entry[f"core_{field}"] != entry[f"public_{field}"]:
                raise AssertionError(
                    f"Adapter changed numerical work: {problem}/{solver}"
                )
        entry["interface_difference_ms"] = (
            entry["public_median_ms"] - entry["core_median_ms"]
        )
        entry["public_over_core"] = entry["public_median_ms"] / entry["core_median_ms"]
        summary.append(entry)
    return summary


def run_benchmark(output, repetitions=11, batch_size=3):
    """Interleave direct/public batches on three 100-variable problems and save evidence."""
    if repetitions < 3 or repetitions % 2 != 1 or batch_size < 1:
        raise ValueError("Use odd repetitions >= 3 and batch_size >= 1.")
    output.mkdir(parents=True, exist_ok=True)
    configurations = [
        (problem, solver, timing.make_options(problem, solver))
        for problem in timing.get_large_problems()
        for solver in timing.CUSTOM_SOLVERS + timing.CONTROLLED_SOLVERS
    ]
    for problem, solver, options in configurations:
        for interface in ("core", "public"):
            for _ in range(2):
                result = timing.run_solver(
                    problem,
                    solver,
                    options,
                    problem.residual,
                    problem.gradient,
                    interface=interface,
                )
                timing.check_result(problem, solver, options, result)
    rows = []
    randomizer = random.Random(20261025)
    for repetition in range(repetitions):
        jobs = [
            (configuration, interface, measured)
            for configuration in configurations
            for interface in ("core", "public")
            for measured in (False, True)
        ]
        randomizer.shuffle(jobs)
        for (problem, solver, options), interface, measured in jobs:
            row, _ = timing.measure_batch(
                problem, solver, options, batch_size, measured, interface=interface
            )
            row.update(interface=interface, repetition=repetition)
            rows.append(row)
        print(f"Completed interface round {repetition + 1}/{repetitions}", flush=True)
    summary = summarize(rows)
    timing.write_csv(output / "raw_batches.csv", rows)
    timing.write_csv(output / "summary.csv", summary)
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        repetitions=repetitions,
        batch_size=batch_size,
        warmups_per_interface=2,
        residual_target=timing.TARGET,
        timed_solves=len(configurations) * repetitions * batch_size * 4,
        numpy=np.__version__,
        scipy=timing.scipy.__version__,
        python=timing.platform.python_version(),
        platform=timing.platform.platform(),
        thread_environment={
            key: os.environ.get(key) for key in timing.THREAD_VARIABLES
        },
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    lines = [
        "# Public root interface timing",
        "",
        "All workloads have 100 variables and analytic Jacobians. Both interfaces retain full histories. "
        "Public timing includes construction, option resolution, accepted-point recording, result copies, "
        "and reporting preparation with printing disabled. Direct timing uses the same packaged core. "
        "Residual checks, warmups, imports, and problem construction are outside timing.",
        "",
        "Differences of clean medians estimate total interface cost, including measurement noise; "
        "they are not isolated linear-algebra timings. CSV components come from one median-total "
        "instrumented batch per interface and add to that batch's total. Quartiles and raw batches "
        "are retained. Counts must agree between interfaces.",
        "",
        "| Problem | Solver / policy | Core ms | Public ms | Difference ms | Ratio | Roots verified |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        config = timing.solver_configuration(row["solver"])
        lines.append(
            f"| {row['problem']} | {timing.METHOD_NAMES[config.method]} / {config.policy} | "
            f"{row['core_median_ms']:.3f} | {row['public_median_ms']:.3f} | "
            f"{row['interface_difference_ms']:.3f} | {row['public_over_core']:.2f} | {row['all_root_success']} |"
        )
    lines += [
        "",
        "See [summary.csv](summary.csv), [raw_batches.csv](raw_batches.csv), and [metadata.json](metadata.json).",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved interface timings to {output.resolve()}")


def main():
    """Parse output and repetition settings for the interface-only timing study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results_adapter_perf"
    )
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.output, args.repetitions, args.batch_size)


if __name__ == "__main__":
    main()

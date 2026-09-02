#!/usr/bin/env python3

"""
Quantum ESPRESSO execution helper.

Runs prepared QE calculations sequentially and captures
stdout/stderr for each calculation.

Example:

    python qe_runner.py references/reference_runs/*/*

Run each calculation with 6 MPI processes:

    python qe_runner.py references/reference_runs/*/* --mpi 6

Test without actually running QE:

    python qe_runner.py references/reference_runs/*/* --mpi 6 --dry-run

By default, calculations are run SEQUENTIALLY:

    H2 -> finish
    N2 -> finish
    O2 -> finish
    Na -> finish

MPI controls how many processes are used for each
individual calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import argparse
import json
import subprocess
import time


# --------------------------------------------------------
# DATA CLASSES
# --------------------------------------------------------

@dataclass(frozen=True)
class QERunnerSettings:
    """Settings for launching a QE calculation."""

    pw_executable: str = "pw.x"

    input_filename: str = "qe.in"

    output_filename: str = "qe.out"

    stderr_filename: str = "qe.err"

    stdin_redirection: bool = True

    mpi_command: Optional[Sequence[str]] = None

    dry_run: bool = False

    rerun_completed: bool = False


@dataclass(frozen=True)
class QERunResult:
    """Summary of a QE run."""

    folder: Path

    returncode: int

    output_path: Path

    error_path: Path

    status: str

    elapsed_seconds: float


# --------------------------------------------------------
# BUILD QE COMMAND
# --------------------------------------------------------

def build_command(
    settings: QERunnerSettings,
) -> list[str]:
    """
    Construct the command used to launch QE.

    Examples:

        pw.x

    or:

        mpirun -np 6 pw.x
    """

    cmd: list[str] = []

    if settings.mpi_command:

        cmd.extend(
            settings.mpi_command
        )

    cmd.append(
        settings.pw_executable
    )

    return cmd


# --------------------------------------------------------
# CHECK WHETHER QE FINISHED
# --------------------------------------------------------

def job_done(
    output_path: Path,
) -> bool:
    """
    Check for Quantum ESPRESSO's normal completion
    marker:

        JOB DONE.
    """

    if not output_path.exists():

        return False

    try:

        with open(
            output_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            for line in f:

                if "JOB DONE." in line:

                    return True

    except OSError:

        return False

    return False


# --------------------------------------------------------
# FORMAT ELAPSED TIME
# --------------------------------------------------------

def format_time(
    seconds: float,
) -> str:
    """Convert seconds to HH:MM:SS."""

    seconds = int(seconds)

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# --------------------------------------------------------
# RUN ONE QE CALCULATION
# --------------------------------------------------------

def run_qe_folder(
    folder: str | Path,
    settings: QERunnerSettings = QERunnerSettings(),
) -> QERunResult:
    """
    Run QE in one prepared calculation folder.
    """

    folder = Path(
        folder
    ).resolve()


    # ----------------------------------------------------
    # CHECK DIRECTORY
    # ----------------------------------------------------

    if not folder.is_dir():

        raise NotADirectoryError(
            f"Not a directory: {folder}"
        )


    # ----------------------------------------------------
    # FILE PATHS
    # ----------------------------------------------------

    input_path = (
        folder
        / settings.input_filename
    )

    output_path = (
        folder
        / settings.output_filename
    )

    error_path = (
        folder
        / settings.stderr_filename
    )


    # ----------------------------------------------------
    # CHECK INPUT FILE
    # ----------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"Missing QE input file:\n"
            f"  {input_path}"
        )


    # ----------------------------------------------------
    # CHECK FOR ALREADY COMPLETED JOB
    # ----------------------------------------------------

    if (
        job_done(output_path)
        and not settings.rerun_completed
    ):

        print(
            "  Already completed."
        )

        print(
            "  Found JOB DONE. "
            "Skipping calculation."
        )

        return QERunResult(
            folder=folder,
            returncode=0,
            output_path=output_path,
            error_path=error_path,
            status="already_completed",
            elapsed_seconds=0.0,
        )


    # ----------------------------------------------------
    # BUILD COMMAND
    # ----------------------------------------------------

    cmd = build_command(
        settings
    )


    # ----------------------------------------------------
    # DRY RUN
    # ----------------------------------------------------

    if settings.dry_run:

        meta = {

            "command":
                cmd,

            "folder":
                str(folder),

            "input":
                str(input_path),

            "output":
                str(output_path),

            "stderr":
                str(error_path),

        }


        dry_file = (
            folder
            / "dry_run_command.json"
        )


        with open(
            dry_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                meta,
                f,
                indent=2,
                sort_keys=True,
            )


        print(
            "  DRY RUN"
        )

        print(
            "  Command:"
        )

        print(
            "  "
            + " ".join(cmd)
        )

        return QERunResult(
            folder=folder,
            returncode=0,
            output_path=output_path,
            error_path=error_path,
            status="dry_run",
            elapsed_seconds=0.0,
        )


    # ----------------------------------------------------
    # RUN QE
    # ----------------------------------------------------

    print(
        "  Command:"
    )

    print(
        "  "
        + " ".join(cmd)
    )

    print(
        "  Running..."
    )


    start_time = time.time()


    with (
        open(
            input_path,
            "r",
            encoding="utf-8",
        ) as fin,

        open(
            output_path,
            "w",
            encoding="utf-8",
        ) as fout,

        open(
            error_path,
            "w",
            encoding="utf-8",
        ) as ferr
    ):

        completed = subprocess.run(

            cmd,

            stdin=(
                fin
                if settings.stdin_redirection
                else None
            ),

            stdout=fout,

            stderr=ferr,

            cwd=str(folder),

            check=False,

            text=True,
        )


    elapsed = (
        time.time()
        - start_time
    )


    # ----------------------------------------------------
    # DETERMINE STATUS
    # ----------------------------------------------------

    if completed.returncode != 0:

        status = "failed"

    elif job_done(output_path):

        status = "completed"

    else:

        status = "incomplete"


    return QERunResult(

        folder=folder,

        returncode=int(
            completed.returncode
        ),

        output_path=output_path,

        error_path=error_path,

        status=status,

        elapsed_seconds=elapsed,

    )


# --------------------------------------------------------
# RUN MULTIPLE CALCULATIONS
# --------------------------------------------------------

def run_many_qe_folders(
    folders: Sequence[str | Path],
    settings: QERunnerSettings = QERunnerSettings(),
) -> list[QERunResult]:
    """
    Run QE calculations sequentially.

    Only ONE calculation is launched at a time.

    If MPI is enabled, each individual calculation
    may use multiple MPI processes.
    """

    results: list[QERunResult] = []

    total = len(
        folders
    )


    for index, folder in enumerate(
        folders,
        start=1,
    ):

        print()
        print(
            "========================================"
        )

        print(
            f"Calculation {index} of {total}"
        )

        print(
            "========================================"
        )

        print(
            f"Folder:"
        )

        print(
            f"  {folder}"
        )

        print()


        try:

            result = run_qe_folder(
                folder,
                settings=settings,
            )


        except Exception as exc:

            print(
                f"ERROR:"
            )

            print(
                f"  {exc}"
            )

            result = QERunResult(

                folder=Path(
                    folder
                ),

                returncode=-1,

                output_path=(
                    Path(folder)
                    / settings.output_filename
                ),

                error_path=(
                    Path(folder)
                    / settings.stderr_filename
                ),

                status="error",

                elapsed_seconds=0.0,
            )


        results.append(
            result
        )


        # ------------------------------------------------
        # REPORT RESULT
        # ------------------------------------------------

        print()

        print(
            f"Status: "
            f"{result.status}"
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        print(
            f"Elapsed: "
            f"{format_time(result.elapsed_seconds)}"
        )


    return results


# --------------------------------------------------------
# COMMAND LINE
# --------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(
            "Run Quantum ESPRESSO calculations "
            "in prepared folders."
        )

    )


    # ----------------------------------------------------
    # CALCULATION FOLDERS
    # ----------------------------------------------------

    parser.add_argument(

        "folders",

        nargs="+",

        help=(
            "QE calculation folders to run."
        ),

    )


    # ----------------------------------------------------
    # PW EXECUTABLE
    # ----------------------------------------------------

    parser.add_argument(

        "--pw",

        dest="pw_executable",

        default="pw.x",

        help=(
            "Quantum ESPRESSO pw.x executable. "
            "Default: pw.x"
        ),

    )


    # ----------------------------------------------------
    # MPI
    # ----------------------------------------------------

    parser.add_argument(

        "--mpi",

        type=int,

        default=None,

        metavar="N",

        help=(
            "Run each QE calculation with N MPI "
            "processes. Example: --mpi 6"
        ),

    )


    # ----------------------------------------------------
    # DRY RUN
    # ----------------------------------------------------

    parser.add_argument(

        "--dry-run",

        action="store_true",

        help=(
            "Show/write commands without "
            "actually running QE."
        ),

    )


    # ----------------------------------------------------
    # FORCE RERUN
    # ----------------------------------------------------

    parser.add_argument(

        "--rerun",

        action="store_true",

        help=(
            "Rerun calculations even when "
            "qe.out already contains "
            "'JOB DONE.'."
        ),

    )


    args = parser.parse_args()


    # ----------------------------------------------------
    # VALIDATE MPI
    # ----------------------------------------------------

    mpi_command = None


    if args.mpi is not None:

        if args.mpi < 1:

            parser.error(
                "--mpi must be at least 1."
            )


        mpi_command = [

            "mpirun",

            "-np",

            str(args.mpi),

        ]


    # ----------------------------------------------------
    # SETTINGS
    # ----------------------------------------------------

    settings = QERunnerSettings(

        pw_executable=
            args.pw_executable,

        mpi_command=
            mpi_command,

        dry_run=
            args.dry_run,

        rerun_completed=
            args.rerun,

    )


    # ----------------------------------------------------
    # STARTUP INFORMATION
    # ----------------------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        "Quantum ESPRESSO Runner"
    )

    print(
        "========================================"
    )

    print()


    print(
        f"Calculations: "
        f"{len(args.folders)}"
    )


    if args.mpi is None:

        print(
            "MPI:          disabled"
        )

    else:

        print(
            f"MPI:          "
            f"{args.mpi} processes per calculation"
        )


    print(
        "Execution:    sequential"
    )


    if args.dry_run:

        print(
            "Mode:         DRY RUN"
        )

    else:

        print(
            "Mode:         RUN"
        )


    print()


    # ----------------------------------------------------
    # RUN
    # ----------------------------------------------------

    total_start = (
        time.time()
    )


    results = run_many_qe_folders(

        args.folders,

        settings=settings,

    )


    total_elapsed = (
        time.time()
        - total_start
    )


    # ----------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------

    print()
    print()
    print(
        "========================================"
    )

    print(
        "QE Run Summary"
    )

    print(
        "========================================"
    )

    print()


    for result in results:

        print(

            f"{result.folder.name:30s} "
            f"{result.status:20s} "
            f"{format_time(result.elapsed_seconds)}"

        )


    # ----------------------------------------------------
    # COUNTS
    # ----------------------------------------------------

    completed = sum(

        result.status == "completed"

        for result in results

    )


    already_completed = sum(

        result.status == "already_completed"

        for result in results

    )


    failed = sum(

        result.status in {
            "failed",
            "incomplete",
            "error",
        }

        for result in results

    )


    print()
    print(
        "----------------------------------------"
    )

    print(
        f"Completed:         {completed}"
    )

    print(
        f"Already completed: {already_completed}"
    )

    print(
        f"Failed/incomplete: {failed}"
    )

    print(
        f"Total time:        "
        f"{format_time(total_elapsed)}"
    )

    print()


    if failed == 0:

        print(
            "All calculations completed successfully."
        )

    else:

        print(
            "One or more calculations did not "
            "complete successfully."
        )

    print()

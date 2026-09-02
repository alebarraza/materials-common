"""Parse Quantum ESPRESSO relaxation outputs.

This module extracts information from a completed QE relaxation:

1. final total energy in Ry and eV
2. final relaxed structure as an ASE Atoms object
3. convergence status
4. optional CIF output
5. optional CSV summary

Both fixed-cell ``relax`` and variable-cell ``vc-relax`` calculations
are supported.

For ``vc-relax``, the final CELL_PARAMETERS block is used.

For fixed-cell ``relax``, QE may not print CELL_PARAMETERS together with
the final coordinates. In that case, the simulation cell is recovered
from the initial ``crystal axes`` block printed by QE.

Supported QE coordinate/cell units include:
    angstrom
    bohr
    alat
    crystal   (ATOMIC_POSITIONS only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import re

import numpy as np
from ase import Atoms
from ase.units import Bohr


RY_TO_EV = 13.605693122994


@dataclass(frozen=True)
class QEResult:
    """Parsed summary of a QE relaxation output."""

    energy_ry: Optional[float]
    energy_ev: Optional[float]
    atoms: Optional[Atoms]
    converged: bool
    source: Path


# ----------------------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------------------

_energy_patterns = [
    re.compile(
        r"!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry",
        re.IGNORECASE,
    ),
    re.compile(
        r"total energy\s+=\s+([-+0-9.Ee]+)\s+Ry",
        re.IGNORECASE,
    ),
]

_cell_start = re.compile(
    r"^\s*CELL_PARAMETERS(?:\s*\(([^)]+)\))?\s*$",
    re.IGNORECASE,
)

_pos_start = re.compile(
    r"^\s*ATOMIC_POSITIONS(?:\s*\(([^)]+)\))?\s*$",
    re.IGNORECASE,
)

_end_marker = re.compile(
    r"^\s*(?:"
    r"End of|"
    r"End final coordinates|"
    r"CELL_PARAMETERS|"
    r"ATOMIC_POSITIONS|"
    r"!|"
    r"number of|"
    r"Forces acting on atoms|"
    r"JOB DONE\."
    r")",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Energy / convergence
# ----------------------------------------------------------------------

def _parse_energy(text: str) -> Optional[float]:
    """Return the final total energy in Ry."""

    for pat in _energy_patterns:
        matches = pat.findall(text)

        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    return None


def _parse_converged(text: str) -> bool:
    """Determine whether QE appears to have completed/converged."""

    markers = [
        "JOB DONE.",
        "convergence has been achieved",
        "Final scf step",
        "Begin final coordinates",
    ]

    text_lower = text.lower()

    return any(marker.lower() in text_lower for marker in markers)


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------

def _collect_blocks(
    lines: list[str],
    start_re: re.Pattern[str],
) -> list[tuple[int, str]]:
    """Return indices and headers for matching QE blocks."""

    blocks: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        if start_re.match(line):
            blocks.append((i, line))

    return blocks


def _parse_alat_bohr(text: str) -> Optional[float]:
    """Parse QE's lattice parameter alat in Bohr."""

    patterns = [
        re.compile(
            r"lattice parameter\s+\(alat\)\s*=\s*"
            r"([-+0-9.Ee]+)\s+a\.u\.",
            re.IGNORECASE,
        ),
        re.compile(
            r"celldm\(1\)\s*=\s*([-+0-9.Ee]+)",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(text)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None


def _normalize_units(units: Optional[str]) -> Optional[str]:
    """Normalize a QE unit label."""

    if units is None:
        return None

    units = units.strip().lower()

    # QE output can occasionally include things like:
    # CELL_PARAMETERS (alat= 10.2)
    if units.startswith("alat"):
        return "alat"

    if units.startswith("angstrom"):
        return "angstrom"

    if units.startswith("bohr"):
        return "bohr"

    if units.startswith("crystal"):
        return "crystal"

    return units


# ----------------------------------------------------------------------
# Cell parsing
# ----------------------------------------------------------------------

def _parse_cell_matrix(
    lines: list[str],
    start_index: int,
) -> tuple[np.ndarray, int]:
    """Read the three vectors following CELL_PARAMETERS."""

    cell: list[list[float]] = []

    i = start_index + 1

    while i < len(lines) and len(cell) < 3:
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        parts = line.split()

        if len(parts) < 3:
            break

        try:
            vector = [
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
            ]
        except ValueError:
            break

        cell.append(vector)
        i += 1

    if len(cell) != 3:
        raise ValueError(
            "Could not parse CELL_PARAMETERS block."
        )

    return np.array(cell, dtype=float), i


def _convert_cell_to_angstrom(
    cell: np.ndarray,
    units: Optional[str],
    text: str,
) -> np.ndarray:
    """Convert a QE CELL_PARAMETERS matrix to Angstrom."""

    units = _normalize_units(units)

    if units == "angstrom":
        return cell

    if units == "bohr":
        return cell * Bohr

    if units == "alat" or units is None:
        alat_bohr = _parse_alat_bohr(text)

        if alat_bohr is None:
            raise ValueError(
                "CELL_PARAMETERS uses alat units, but alat "
                "could not be determined."
            )

        return cell * alat_bohr * Bohr

    raise ValueError(
        f"Unsupported CELL_PARAMETERS units: {units!r}"
    )


def _parse_initial_cell(text: str) -> Optional[np.ndarray]:
    """Recover the initial QE cell from the crystal-axes block.

    QE prints something like:

        lattice parameter (alat) = 18.8973 a.u.

        crystal axes: (cart. coord. in units of alat)
             a(1) = ( 1.000000 0.000000 0.000000 )
             a(2) = ( 0.000000 1.000000 0.000000 )
             a(3) = ( 0.000000 0.000000 1.073717 )

    The axis vectors are dimensionless multiples of alat.
    """

    alat_bohr = _parse_alat_bohr(text)

    if alat_bohr is None:
        return None

    axes_pattern = re.compile(
        r"crystal axes:\s*"
        r"\(cart\.\s*coord\.\s*in units of alat\)"
        r".*?"
        r"a\(1\)\s*=\s*\(\s*"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s*\)"
        r".*?"
        r"a\(2\)\s*=\s*\(\s*"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s*\)"
        r".*?"
        r"a\(3\)\s*=\s*\(\s*"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    match = axes_pattern.search(text)

    if match is None:
        return None

    try:
        values = [float(x) for x in match.groups()]
    except ValueError:
        return None

    axes = np.array(values, dtype=float).reshape(3, 3)

    # axes are multiples of alat.
    # alat is in Bohr.
    cell_angstrom = axes * alat_bohr * Bohr

    return cell_angstrom


def _find_final_cell(
    text: str,
    lines: list[str],
) -> Optional[np.ndarray]:
    """Find the most appropriate QE cell in Angstrom.

    Priority:
        1. final CELL_PARAMETERS block
        2. initial crystal-axes block
    """

    cell_blocks = _collect_blocks(lines, _cell_start)

    if cell_blocks:
        cell_index, cell_header = cell_blocks[-1]

        try:
            raw_cell, _ = _parse_cell_matrix(
                lines,
                cell_index,
            )

            match = _cell_start.match(cell_header)

            units = (
                match.group(1)
                if match is not None
                else None
            )

            return _convert_cell_to_angstrom(
                raw_cell,
                units,
                text,
            )

        except ValueError:
            pass

    # Fixed-cell relax fallback
    return _parse_initial_cell(text)


# ----------------------------------------------------------------------
# Atomic positions
# ----------------------------------------------------------------------

def _parse_atomic_positions(
    lines: list[str],
    start_index: int,
) -> tuple[list[str], np.ndarray, int]:
    """Read an ATOMIC_POSITIONS block."""

    symbols: list[str] = []
    positions: list[list[float]] = []

    i = start_index + 1

    while i < len(lines):
        raw = lines[i].strip()

        if not raw:
            i += 1
            continue

        if _end_marker.match(raw):
            break

        parts = raw.split()

        if len(parts) < 4:
            break

        symbol = parts[0]

        try:
            xyz = [
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            ]
        except ValueError:
            break

        symbols.append(symbol)
        positions.append(xyz)

        i += 1

    if not symbols:
        raise ValueError(
            "Could not parse ATOMIC_POSITIONS block."
        )

    return (
        symbols,
        np.array(positions, dtype=float),
        i,
    )


def _convert_positions_to_angstrom(
    positions: np.ndarray,
    units: Optional[str],
    cell: Optional[np.ndarray],
    text: str,
) -> np.ndarray:
    """Convert QE atomic positions to Cartesian Angstrom."""

    units = _normalize_units(units)

    if units == "angstrom":
        return positions

    if units == "bohr":
        return positions * Bohr

    if units == "alat" or units is None:
        alat_bohr = _parse_alat_bohr(text)

        if alat_bohr is None:
            raise ValueError(
                "ATOMIC_POSITIONS uses alat units, but alat "
                "could not be determined."
            )

        return positions * alat_bohr * Bohr

    if units == "crystal":
        if cell is None:
            raise ValueError(
                "ATOMIC_POSITIONS uses crystal coordinates, "
                "but no cell could be determined."
            )

        # Fractional coordinates -> Cartesian coordinates
        return positions @ cell

    raise ValueError(
        f"Unsupported ATOMIC_POSITIONS units: {units!r}"
    )


# ----------------------------------------------------------------------
# Structure parsing
# ----------------------------------------------------------------------

def read_final_structure_from_output(
    path: str | Path,
) -> Atoms:
    """Read the final relaxed geometry from a QE output file."""

    path = Path(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    # --------------------------------------------------------------
    # Find final ATOMIC_POSITIONS
    # --------------------------------------------------------------

    pos_blocks = _collect_blocks(
        lines,
        _pos_start,
    )

    if not pos_blocks:
        raise ValueError(
            "No ATOMIC_POSITIONS block found in QE output."
        )

    pos_index, pos_header = pos_blocks[-1]

    symbols, raw_positions, _ = _parse_atomic_positions(
        lines,
        pos_index,
    )

    # --------------------------------------------------------------
    # Determine cell
    # --------------------------------------------------------------

    cell = _find_final_cell(
        text,
        lines,
    )

    # --------------------------------------------------------------
    # Determine position units
    # --------------------------------------------------------------

    pos_match = _pos_start.match(pos_header)

    pos_units = (
        pos_match.group(1)
        if pos_match is not None
        else None
    )

    positions = _convert_positions_to_angstrom(
        raw_positions,
        pos_units,
        cell,
        text,
    )

    # --------------------------------------------------------------
    # Construct ASE Atoms
    # --------------------------------------------------------------

    if cell is None:
        # No valid simulation cell was recovered.
        #
        # Do NOT set pbc=True here. Doing so creates:
        #
        #     Cell([0, 0, 0]), pbc=True
        #
        # which causes ASE's CIF writer to fail when it attempts
        # to wrap positions.
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            pbc=False,
        )

    else:
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )

    atoms.info["source_output"] = str(path)
    atoms.info["parsed_from"] = "qe_output"

    return atoms


# ----------------------------------------------------------------------
# Public parsing functions
# ----------------------------------------------------------------------

def parse_total_energy(
    path: str | Path,
) -> tuple[Optional[float], Optional[float]]:
    """Return the last total energy in Ry and eV."""

    path = Path(path)

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    energy_ry = _parse_energy(text)

    energy_ev = (
        None
        if energy_ry is None
        else energy_ry * RY_TO_EV
    )

    return energy_ry, energy_ev


def parse_qe_output(
    path: str | Path,
) -> QEResult:
    """Parse a QE relaxation output into a compact summary."""

    path = Path(path)

    energy_ry, energy_ev = parse_total_energy(path)

    atoms = None

    try:
        atoms = read_final_structure_from_output(path)
    except Exception as exc:
        # Keep energy parsing usable even when the structure
        # cannot be recovered.
        print(
            f"Warning: could not parse structure from "
            f"{path}: {exc}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    converged = _parse_converged(text)

    return QEResult(
        energy_ry=energy_ry,
        energy_ev=energy_ev,
        atoms=atoms,
        converged=converged,
        source=path,
    )


# ----------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------

def result_to_row(result: QEResult) -> dict:
    """Convert QEResult into a dictionary suitable for CSV output."""

    atoms = result.atoms

    if atoms is not None:
        formula = atoms.get_chemical_formula()
        natoms = len(atoms)

        # ASE requires a full 3D cell for volume.
        if atoms.cell.rank == 3:
            volume_ang3 = atoms.get_volume()
        else:
            volume_ang3 = None

    else:
        formula = None
        natoms = None
        volume_ang3 = None

    return {
        "source": str(result.source),
        "converged": result.converged,
        "energy_ry": result.energy_ry,
        "energy_ev": result.energy_ev,
        "formula": formula,
        "natoms": natoms,
        "volume_ang3": volume_ang3,
        "has_structure": atoms is not None,
    }


def write_result_csv(
    path: str | Path,
    result: QEResult,
) -> None:
    """Write one parsed QE result to a CSV file.

    If the file does not exist, a header is created.

    If the file already exists, the new result is appended.
    This allows multiple QE calculations to be collected into
    one master CSV.
    """

    path = Path(path)

    # Create parent directories if needed.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = result_to_row(result)

    file_exists = path.exists()

    # If the file exists but is empty, we still need a header.
    file_has_content = (
        file_exists
        and path.stat().st_size > 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if not file_has_content:
            writer.writeheader()

        writer.writerow(row)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ase.io import write

    parser = argparse.ArgumentParser(
        description=(
            "Parse a Quantum ESPRESSO relaxation output."
        )
    )

    parser.add_argument(
        "outputs",
        type=str,
        nargs="+",
        help=(
            "QE output file, e.g. qe_relax.out"
        ),
    )

    parser.add_argument(
        "--write-cif",
        type=str,
        default=None,
        help=(
            "Optional path for writing the final structure "
            "as a CIF file."
        ),
    )

    parser.add_argument(
        "--write-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV path. Results are appended if the "
            "CSV already exists."
        ),
    )

    args = parser.parse_args()

# --------------------------------------------------------------
# Parse all supplied QE output files
# --------------------------------------------------------------

    for output in args.outputs:

        print()
        print("=" * 80)
        print(output)
        print("=" * 80)

        result = parse_qe_output(output)

        # ----------------------------------------------------------
        # Print summary
        # ----------------------------------------------------------

        print(f"converged={result.converged}")
        print(f"energy_ry={result.energy_ry}")
        print(f"energy_ev={result.energy_ev}")
        print(f"has_structure={result.atoms is not None}")

        if result.atoms is not None:

            print(
                f"formula={result.atoms.get_chemical_formula()}"
            )

            print(
                f"natoms={len(result.atoms)}"
            )

            if result.atoms.cell.rank == 3:
                print(
                    f"volume_ang3={result.atoms.get_volume()}"
                )

        # ----------------------------------------------------------
        # CIF output
        # ----------------------------------------------------------

        if args.write_cif and result.atoms is not None:

            # If multiple outputs are being processed, treating
            # --write-cif as a single filename would overwrite the
            # previous structure. Instead, use it as an output
            # directory.
            if len(args.outputs) > 1:

                cif_dir = Path(args.write_cif)

                cif_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cif_path = (
                    cif_dir
                    / f"{Path(output).parent.name}_final.cif"
                )

            else:

                cif_path = Path(args.write_cif)

                cif_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            write(
                cif_path,
                result.atoms,
            )

            print(f"wrote {cif_path}")

        # ----------------------------------------------------------
        # CSV output
        # ----------------------------------------------------------

        if args.write_csv:

            write_result_csv(
                args.write_csv,
                result,
            )

            print(
                f"appended {args.write_csv}"
            )

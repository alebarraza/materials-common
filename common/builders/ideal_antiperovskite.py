"""Builders for ideal sodium antiperovskite structures.

This module is intentionally small and reproducible. It creates the ideal
Na3OCl antiperovskite directly from symmetry, without requiring a CIF file.

Why start here?
- The primitive cell is tiny and easy to validate.
- The structure is fully reproducible from the lattice constant alone.
- The same builder can be reused later for hydroxylated and substituted cells.

This file is only the first step in the workflow. It does not yet perform
hydroxylation, defect creation, or Quantum ESPRESSO input generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass(frozen=True)
class AntiperovskiteSpec:
    """Specification for the ideal Na3OCl antiperovskite primitive cell."""

    a: float
    symbols: tuple[str, ...] = ("O", "Cl", "Na", "Na", "Na")
    scaled_positions: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    )


def build_na3ocl(a: float) -> Atoms:
    """Build the ideal cubic Na3OCl primitive cell.

    Parameters
    ----------
    a:
        Cubic lattice parameter in angstrom.

    Returns
    -------
    ase.Atoms
        Primitive Na3OCl cell with 5 atoms and periodic boundary conditions.
    """

    if a <= 0:
        raise ValueError("Lattice constant 'a' must be positive.")

    spec = AntiperovskiteSpec(a=a)
    atoms = Atoms(
        symbols=list(spec.symbols),
        scaled_positions=list(spec.scaled_positions),
        cell=[a, a, a],
        pbc=True,
    )
    atoms.info["builder"] = "build_na3ocl"
    atoms.info["lattice_constant_angstrom"] = float(a)
    atoms.info["composition"] = "Na3OCl"
    atoms.info["structure_type"] = "ideal_antiperovskite"
    return atoms


def build_na3ocl_supercell(a: float, supercell: Sequence[int]) -> Atoms:
    """Build a Na3OCl supercell from the ideal primitive cell.

    Parameters
    ----------
    a:
        Cubic lattice parameter in angstrom.
    supercell:
        Three integers describing the replication along x, y, z.

    Returns
    -------
    ase.Atoms
        Supercell built from the primitive Na3OCl structure.
    """

    if len(supercell) != 3:
        raise ValueError("supercell must contain exactly three integers, e.g. (2, 2, 2).")

    sc = tuple(int(n) for n in supercell)
    if any(n <= 0 for n in sc):
        raise ValueError("All supercell dimensions must be positive integers.")

    atoms = build_na3ocl(a)
    atoms = atoms.repeat(sc)
    atoms.info["builder"] = "build_na3ocl_supercell"
    atoms.info["supercell"] = sc
    atoms.info["composition"] = f"Na3OCl x {sc[0]}x{sc[1]}x{sc[2]}"
    return atoms


def validate_fractional_positions(atoms: Atoms, tol: float = 1e-8) -> None:
    """Basic sanity check for the ideal primitive cell.

    This is useful as a first regression test to confirm that the structure
    still matches the intended symmetry-based construction.
    """

    frac = atoms.get_scaled_positions(wrap=True)
    expected = np.array(AntiperovskiteSpec(a=1.0).scaled_positions, dtype=float)

    if len(frac) != len(expected):
        raise ValueError(f"Expected {len(expected)} atoms, found {len(frac)}.")

    # The atoms can be permuted by downstream code, so we compare against the
    # exact constructor order here only for the pristine primitive cell.
    if not np.allclose(frac, expected, atol=tol, rtol=0.0):
        raise ValueError(
            "Fractional coordinates do not match the expected ideal antiperovskite cell."
        )


def save_structure(atoms: Atoms, path: str | Path) -> None:
    """Save a structure to disk.

    The file type is inferred from the extension.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), atoms)


def load_structure(path: str | Path) -> Atoms:
    """Load a structure from disk."""

    return read(str(path))


def structure_summary(atoms: Atoms) -> str:
    """Return a short human-readable summary for logging."""

    formula = atoms.get_chemical_formula(mode="hill")
    cell = atoms.cell.lengths()
    pbc = atoms.pbc.tolist()
    return (
        f"formula={formula}, natoms={len(atoms)}, "
        f"cell=({cell[0]:.4f}, {cell[1]:.4f}, {cell[2]:.4f}) Å, pbc={pbc}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the ideal Na3OCl antiperovskite.")
    parser.add_argument("--a", type=float, required=True, help="Lattice parameter in angstrom.")
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Optional supercell replication, e.g. --supercell 2 2 2",
    )
    parser.add_argument("--out", type=str, default="na3ocl.cif", help="Output file path.")
    args = parser.parse_args()

    if args.supercell is None:
        atoms = build_na3ocl(args.a)
    else:
        atoms = build_na3ocl_supercell(args.a, args.supercell)

    save_structure(atoms, args.out)
    print(structure_summary(atoms))


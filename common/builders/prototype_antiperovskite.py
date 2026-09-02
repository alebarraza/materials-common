"""
Builders for ideal cubic A3BX antiperovskites.

These builders create the ideal cubic Pm-3m antiperovskite structure
from crystallographic prototype information only.

Examples
--------
Na3OCl
Li3OCl
Na3BrO
...
"""

from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms


@dataclass(frozen=True)
class PrototypeAntiperovskite:

    A: str
    B: str
    X: str
    a: float


def build_prototype(spec: PrototypeAntiperovskite) -> Atoms:
    """Build an ideal cubic A3BX antiperovskite."""

    if spec.a <= 0:
        raise ValueError("Lattice parameter must be positive.")

    symbols = (
        spec.B,
        spec.X,
        spec.A,
        spec.A,
        spec.A,
    )

    positions = (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    )

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=positions,
        cell=[spec.a] * 3,
        pbc=True,
    )

    atoms.info.update(
        {
            "prototype": "A3BX",
            "spacegroup": "Pm-3m",
            "builder": "prototype_antiperovskite",
        }
    )

    return atoms

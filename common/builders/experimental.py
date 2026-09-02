"""
Build crystal structures directly from experimental crystallographic data.

This builder reconstructs a full crystal structure from an experimental
asymmetric unit using the reported space group and lattice parameters.

The resulting object is returned as an ASE Atoms instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor


@dataclass(frozen=True)
class AtomSite:
    element: str
    position: tuple[float, float, float]
    label: str | None = None
    wyckoff: str | None = None
    occupancy: float = 1.0


@dataclass(frozen=True)
class ExperimentalStructure:
    spacegroup: str
    lattice: dict
    atoms: list[AtomSite]
    reference: dict


class ExperimentalBuilder:

    def from_yaml(self, filename: str | Path):

        filename = Path(filename)

        with open(filename) as f:
            data = yaml.safe_load(f)

        lattice = Lattice.from_parameters(
            data["lattice"]["a"],
            data["lattice"]["b"],
            data["lattice"]["c"],
            data["lattice"].get("alpha", 90),
            data["lattice"].get("beta", 90),
            data["lattice"].get("gamma", 90),
        )

        species = []

        coords = []

        for atom in data["atoms"]:

            species.append(atom["element"])

            coords.append(atom["position"])

        structure = Structure.from_spacegroup(
            sg=data["spacegroup"]["symbol"],
            lattice=lattice,
            species=species,
            coords=coords,
        )

        atoms = AseAtomsAdaptor.get_atoms(structure)

        atoms.info["reference"] = data["reference"]

        atoms.info["spacegroup"] = data["spacegroup"]

        return atoms

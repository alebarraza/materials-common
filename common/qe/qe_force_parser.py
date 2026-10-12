from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from ase import Atoms


FORCE_PATTERN = re.compile(
    r"atom\s+(\d+)\s+type\s+(\d+)\s+force\s*=\s*"
    r"([-\d.Ee]+)\s+"
    r"([-\d.Ee]+)\s+"
    r"([-\d.Ee]+)"
)

def parse_force_block(
    outfile: str | Path,
    atoms: Atoms,
) -> pd.DataFrame:

    outfile = Path(outfile)

    lines = outfile.read_text().splitlines()

    start = None
    end = None

    for i, line in enumerate(lines):

        if "Forces acting on atoms" in line:
            start = i

        elif (
            start is not None
            and "The non-local contrib." in line
        ):
            end = i
            break

    if start is None:
        raise RuntimeError(
            f"No force block found in {outfile}"
        )

    block = "\n".join(lines[start:end])

    rows = []

    for atom, _, fx, fy, fz in FORCE_PATTERN.findall(block):

        atom = int(atom)

        fx = float(fx)
        fy = float(fy)
        fz = float(fz)

        symbol = atoms[atom - 1].symbol

        rows.append({

            "Atom": atom,

            "Label": f"{symbol}{atom}",

            "Element": symbol,

            "Fx": fx,

            "Fy": fy,

            "Fz": fz,

            "Magnitude":
                np.linalg.norm([fx, fy, fz]),

        })

    return pd.DataFrame(rows)

def parse_stress(
    outfile: str | Path,
) -> dict:

    outfile = Path(outfile)

    lines = outfile.read_text().splitlines()

    for i, line in enumerate(lines):

        if "total   stress" not in line:
            continue

        pressure = float(
            line.split("P=")[1]
        )

        row1 = lines[i + 1].split()
        row2 = lines[i + 2].split()
        row3 = lines[i + 3].split()

        return {

            "P":
                pressure,

            "sigma_xx":
                float(row1[3]),

            "sigma_yy":
                float(row2[4]),

            "sigma_zz":
                float(row3[5]),

        }

    raise RuntimeError(
        f"No stress tensor found in {outfile}"
    )

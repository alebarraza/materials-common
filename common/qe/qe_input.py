#!/usr/bin/env python3

"""
Quantum ESPRESSO input preparation utilities.

Supports:

    scf
    relax
    vc-relax
    md

This module can be imported by other Python scripts:

    from qe_relaxation_complete import (
        QEInputSettings,
        write_relax_input,
    )

The command-line interface is protected by:

    if __name__ == "__main__":

so importing this module will NOT execute argparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import argparse
import json

import numpy as np

from ase import Atoms
from ase.io import read
from ase.io.espresso import write_espresso_in


# ============================================================
# PROJECT PATHS
# ============================================================

DEFAULT_PSEUDO_DIR = None


# ============================================================
# QE SETTINGS
# ============================================================

@dataclass(frozen=True)
class QEInputSettings:
    """
    Settings for Quantum ESPRESSO calculations.

    Supported calculation types:

        scf
        relax
        vc-relax
        md
    """

    # --------------------------------------------------------
    # CONTROL
    # --------------------------------------------------------

    calculation: str = "vc-relax"

    prefix: str = None

    pseudo_dir: str | None = None

    def __post_init__(self):
        if self.pseudo_dir is None:
            raise ValueError("pseudo_dir must be specified.")

    verbosity: str = "high"

    tstress: bool = True

    tprnfor: bool = True


    # --------------------------------------------------------
    # PLANE-WAVE CUTOFFS
    # --------------------------------------------------------

    ecutwfc: float = 50.0

    ecutrho: float = 400.0


    # --------------------------------------------------------
    # K-POINTS
    # --------------------------------------------------------

    kpts: tuple[int, int, int] = (
        3,
        3,
        3,
    )

    koffset: tuple[int, int, int] = (
        0,
        0,
        0,
    )


    # --------------------------------------------------------
    # ELECTRONIC CONVERGENCE
    # --------------------------------------------------------

    # Goes into &ELECTRONS
    conv_thr: float = 1.0e-8


    # --------------------------------------------------------
    # IONIC RELAXATION CONVERGENCE
    # --------------------------------------------------------

    # Goes into &CONTROL
    etot_conv_thr: Optional[float] = None

    forc_conv_thr: float = 1.0e-4


    # --------------------------------------------------------
    # CELL RELAXATION
    # --------------------------------------------------------

    press_conv_thr: Optional[float] = None

    cell_dynamics: Optional[str] = None

    trust_radius_min: Optional[float] = None

    trust_radius_max: Optional[float] = None


    # --------------------------------------------------------
    # SYMMETRY
    # --------------------------------------------------------

    use_symmetry: bool = False


    # --------------------------------------------------------
    # SPIN / OCCUPATIONS
    # --------------------------------------------------------

    nspin: int = 1

    occupations: Optional[str] = None

    smearing: Optional[str] = None

    degauss: Optional[float] = None


    # --------------------------------------------------------
    # IONIC DYNAMICS
    # --------------------------------------------------------

    ion_dynamics: str = "bfgs"

    ion_velocities: Optional[str] = None


    # --------------------------------------------------------
    # MOLECULAR DYNAMICS
    # --------------------------------------------------------

    # Number of MD steps
    nstep: Optional[int] = None

    # QE ionic timestep in atomic units
    dt: Optional[float] = None

    # Examples:
    #
    #   "not_controlled"
    #   "rescaling"
    #   "rescale-v"
    #   "rescale-T"
    #   "reduce-T"
    #   "berendsen"
    #   "andersen"
    #
    ion_temperature: Optional[str] = None

    tempw: Optional[float] = None

    tolp: Optional[float] = None

    delta_t: Optional[float] = None


    # --------------------------------------------------------
    # EXTRA QE PARAMETERS
    # --------------------------------------------------------

    extra_system: Mapping[str, object] = field(
        default_factory=dict
    )

    extra_control: Mapping[str, object] = field(
        default_factory=dict
    )

    extra_electrons: Mapping[str, object] = field(
        default_factory=dict
    )

    extra_cell: Mapping[str, object] = field(
        default_factory=dict
    )

    extra_ions: Mapping[str, object] = field(
        default_factory=dict
    )

from dataclasses import dataclass

@dataclass(frozen=True)
class Protocol:
    """
    Reproducible computational protocol.

    A Protocol defines the computational methodology used
    for one study. It wraps QEInputSettings together with
    metadata such as the exchange-correlation functional
    and pseudopotential library.

    This class is not tied to Quantum ESPRESSO input
    generation. Instead it documents the computational
    methodology for reproducibility.
    """

    name: str

    description: str

    functional: str

    pseudo_library: str

    settings: QEInputSettings


# ============================================================
# DEFAULT PSEUDOPOTENTIALS
# ============================================================

DEFAULT_PSEUDOS = {}


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(obj):
    """
    Convert common Python/NumPy/Path objects into forms
    that json.dump() can serialize.
    """

    if isinstance(obj, Path):

        return str(obj)


    if isinstance(obj, np.ndarray):

        return obj.tolist()


    if isinstance(obj, dict):

        return {

            str(key):
                make_json_serializable(value)

            for key, value
            in obj.items()

        }


    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):

        return [

            make_json_serializable(value)

            for value in obj

        ]


    try:

        json.dumps(obj)

        return obj

    except TypeError:

        return str(obj)


# ============================================================
# STRUCTURE HELPERS
# ============================================================

def _ensure_atoms(
    obj: Atoms | str | Path,
) -> Atoms:
    """
    Return a copy of an ASE Atoms object or read a
    structure from disk.
    """

    if isinstance(
        obj,
        Atoms,
    ):

        return obj.copy()


    return read(
        str(obj)
    )


# ============================================================
# PSEUDOPOTENTIAL VALIDATION
# ============================================================

def _validate_pseudos(
    atoms: Atoms,
    pseudos: Mapping[str, str],
) -> None:
    """
    Check that every chemical species has a corresponding
    pseudopotential entry.
    """

    species = set(
        atoms.get_chemical_symbols()
    )


    missing = sorted(

        symbol

        for symbol in species

        if symbol not in pseudos

    )


    if missing:

        raise ValueError(

            "Missing pseudopotentials for species: "
            f"{missing}. "

            "Available keys: "
            f"{sorted(pseudos.keys())}"

        )

# ============================================================
# AIMD INITIAL ATOMIC VELOCITIES
# ============================================================

def _append_atomic_velocities(
    input_path: Path,
    atoms: Atoms,
) -> None:
    """
    Append an ATOMIC_VELOCITIES card to a QE input.

    ASE velocities are returned in Angstrom/fs.

    Quantum ESPRESSO ATOMIC_VELOCITIES values are
    Cartesian velocities in atomic units:
        bohr / atomic_time_unit
    """

    velocities = atoms.get_velocities()

    if velocities is None:
        raise ValueError(
            "Atoms object has no velocities."
        )

    if len(velocities) != len(atoms):
        raise ValueError(
            "Velocity array length does not match atom count."
        )

    # --------------------------------------------------------
    # Unit conversion
    #
    # ASE: Angstrom / fs
    #
    # QE atomic velocity:
    # bohr / atomic unit of time
    #
    # 1 bohr = 0.529177210903 Angstrom
    # 1 atomic time unit = 0.024188843265857 fs
    #
    # v_QE =
    # v_ASE * (atomic_time_fs / bohr_ang)
    # --------------------------------------------------------

    BOHR_ANG = 0.529177210903
    AU_TIME_FS = 0.024188843265857

    conversion = (
        AU_TIME_FS / BOHR_ANG
    )

    velocities_au = (
        velocities * conversion
    )

    with open(
        input_path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write("\n")
        f.write("ATOMIC_VELOCITIES\n")

        for atom, velocity in zip(
            atoms,
            velocities_au,
        ):

            vx, vy, vz = velocity

            f.write(
                f"{atom.symbol:2s} "
                f"{vx: .12e} "
                f"{vy: .12e} "
                f"{vz: .12e}\n"
            )

# ============================================================
# DIRECTORY NAME
# ============================================================

def _directory_name(
    atoms: Atoms,
    fallback: str,
) -> str:
    """
    Construct calculation-directory name.
    """

    formula = atoms.get_chemical_formula(
        mode="hill"
    )

    formula = formula.replace(
        " ",
        "",
    )


    if fallback:

        return (
            f"{fallback}_{formula}"
        )


    return formula


# ============================================================
# QE INPUT DATA
# ============================================================

def _make_input_data(
    settings: QEInputSettings,
) -> dict:
    """
    Convert QERInputSettings into the dictionaries expected
    by ASE's write_espresso_in().
    """

    valid_calculations = {

        "scf",
        "relax",
        "vc-relax",
        "md",

    }


    if settings.calculation not in valid_calculations:

        raise ValueError(

            "Unsupported QE calculation: "
            f"{settings.calculation!r}. "

            "Expected one of: "
            f"{sorted(valid_calculations)}"

        )


    # ========================================================
    # SYSTEM
    # ========================================================

    system = {

        "ecutwfc":
            settings.ecutwfc,

        "ecutrho":
            settings.ecutrho,

        "nspin":
            settings.nspin,

        "nosym":
            not settings.use_symmetry,

        "noinv":
         not settings.use_symmetry,

    }


    if settings.occupations is not None:

        system[
            "occupations"
        ] = settings.occupations


    if settings.smearing is not None:

        system[
            "smearing"
        ] = settings.smearing


    if settings.degauss is not None:

        system[
            "degauss"
        ] = settings.degauss


    system.update(
        dict(
            settings.extra_system
        )
    )


    # ========================================================
    # CONTROL
    # ========================================================

    control = {

        "calculation":
            settings.calculation,

        "prefix":
            settings.prefix,

        "pseudo_dir":
            settings.pseudo_dir,

        "verbosity":
            settings.verbosity,

        "tstress":
            settings.tstress,

        "tprnfor":
            settings.tprnfor,

    }


    # Number of MD steps
    if settings.nstep is not None:

        control[
            "nstep"
        ] = settings.nstep


    # Ionic timestep
    if settings.dt is not None:

        control[
            "dt"
        ] = settings.dt


    # Ionic total-energy convergence.
    #
    # This is separate from electronic conv_thr.

    if settings.etot_conv_thr is not None:

        control[
            "etot_conv_thr"
        ] = settings.etot_conv_thr


    # Force convergence is relevant for structural
    # relaxation, not MD or SCF.

    if settings.calculation in {

        "relax",
        "vc-relax",

    }:

        control[
            "forc_conv_thr"
        ] = settings.forc_conv_thr


    control.update(
        dict(
            settings.extra_control
        )
    )


    # ========================================================
    # ELECTRONS
    # ========================================================

    electrons = {

        "conv_thr":
            settings.conv_thr,

    }


    electrons.update(
        dict(
            settings.extra_electrons
        )
    )


    # ========================================================
    # IONS
    # ========================================================

    ions = {}


    if settings.calculation in {

        "relax",
        "vc-relax",
        "md",

    }:

        ions[
            "ion_dynamics"
        ] = settings.ion_dynamics

    if settings.ion_velocities is not None:
        ions["ion_velocities"] = settings.ion_velocities
    # --------------------------------------------------------
    # MD TEMPERATURE CONTROL
    # --------------------------------------------------------

    if settings.ion_temperature is not None:

        ions[
            "ion_temperature"
        ] = settings.ion_temperature


    if settings.tempw is not None:

        ions[
            "tempw"
        ] = settings.tempw


    if settings.tolp is not None:

        ions[
            "tolp"
        ] = settings.tolp


    if settings.delta_t is not None:

        ions[
            "delta_t"
        ] = settings.delta_t


    ions.update(
        dict(
            settings.extra_ions
        )
    )


    # ========================================================
    # CELL
    # ========================================================

    cell = {}


    if settings.calculation == "vc-relax":

        if settings.cell_dynamics is not None:

            cell[
                "cell_dynamics"
            ] = settings.cell_dynamics


        if settings.press_conv_thr is not None:

            cell[
                "press_conv_thr"
            ] = settings.press_conv_thr


        if settings.trust_radius_min is not None:

            cell[
                "trust_radius_min"
            ] = settings.trust_radius_min


        if settings.trust_radius_max is not None:

            cell[
                "trust_radius_max"
            ] = settings.trust_radius_max


        cell.update(
            dict(
                settings.extra_cell
            )
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "control":
            control,

        "system":
            system,

        "electrons":
            electrons,

        "ions":
            ions,

        "cell":
            cell,

    }


# ============================================================
# WRITE QE INPUT
# ============================================================

def write_qe_input(
    atoms: Atoms,
    outdir: str | Path,
    settings: QEInputSettings | None = None,
    pseudos: Mapping[str, str] = DEFAULT_PSEUDOS,
    kpts: Optional[Sequence[int]] = None,
) -> Path:
    """
    Write a Quantum ESPRESSO input calculation directory.

    Despite the historical function name, this supports:

        scf
        relax
        vc-relax
        md

    Returns
    -------
    Path
        Calculation directory containing qe.in
        and metadata.json.
    """
    if settings is None:
        raise ValueError("settings must be provided.")

    work = atoms.copy()


    # --------------------------------------------------------
    # VALIDATE PSEUDOS
    # --------------------------------------------------------

    _validate_pseudos(
        work,
        pseudos,
    )


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    outdir = Path(
        outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    calc_dir = (

        outdir

        / _directory_name(
            work,
            settings.prefix,
        )

    )


    calc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = _make_input_data(
        settings
    )


    # --------------------------------------------------------
    # K-POINTS
    # --------------------------------------------------------

    kp = (

        tuple(kpts)

        if kpts is not None

        else settings.kpts

    )


    if len(kp) != 3:

        raise ValueError(

            "kpts must be a 3-tuple "
            "such as (1, 1, 1)."

        )


    # --------------------------------------------------------
    # WRITE QE INPUT
    # --------------------------------------------------------

    input_path = (
    calc_dir / "qe.in"
    )

    write_espresso_in(
        str(input_path),
        work,
        input_data=input_data,
        pseudopotentials=dict(pseudos),
        kpts=kp,
        koffset=settings.koffset,
    )


# ------------------------------------------------------------
# MD INITIAL VELOCITIES
# ------------------------------------------------------------

    if (
        settings.calculation == "md"
        and
        settings.ion_velocities == "from_input"
    ):

        velocities = work.get_velocities()

        if velocities is None:

            raise ValueError(
                "ion_velocities='from_input' was requested, "
                "but the Atoms object contains no velocities."
            )

        _append_atomic_velocities(
            input_path,
            work,
        )


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    meta = {

        "builder":
            "write_relax_input",

        "formula":
            work.get_chemical_formula(
                mode="hill"
            ),

        "natoms":
            len(work),

        "cell":
            work.cell.tolist(),

        "pbc":
            work.pbc.tolist(),

        "settings": {

            "calculation":
                settings.calculation,

            "prefix":
                settings.prefix,

            "pseudo_dir":
                settings.pseudo_dir,

            "ecutwfc":
                settings.ecutwfc,

            "ecutrho":
                settings.ecutrho,

            "kpts":
                list(kp),

            "koffset":
                list(
                    settings.koffset
                ),

            "use_symmetry":
                settings.use_symmetry,

            "nspin":
                settings.nspin,

            "conv_thr":
                settings.conv_thr,

            "ion_dynamics":
                settings.ion_dynamics,

            "nstep":
                settings.nstep,

            "dt":
                settings.dt,

            "ion_temperature":
                settings.ion_temperature,

            "tempw":
                settings.tempw,

        },

        "pseudos":
            dict(pseudos),

        "source_info":
            dict(work.info),

    }


    metadata_file = (
        calc_dir
        / "metadata.json"
    )


    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(

            make_json_serializable(
                meta
            ),

            f,

            indent=2,

            sort_keys=True,

        )


    return calc_dir


# ============================================================
# PREPARE MULTIPLE CALCULATIONS
# ============================================================

def prepare_relaxations(
    structures: Iterable[
        Atoms | str | Path
    ],
    outdir: str | Path,
    settings: QEInputSettings | None = None,
    pseudos: Mapping[str, str] = DEFAULT_PSEUDOS,
    start_index: int = 0,
) -> list[Path]:
    """
    Prepare multiple QE calculation folders.
    """
    if settings is None:
        raise ValueError("settings must be provided.")
        outdir = Path(
            outdir
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    created: list[Path] = []


    for i, item in enumerate(
        structures,
        start=start_index,
    ):

        atoms = _ensure_atoms(
            item
        )


        # Dataclass is frozen, so create a new settings
        # object with a candidate-specific prefix.

        settings_dict = dict(
            settings.__dict__
        )


        settings_dict[
            "prefix"
        ] = (

            f"{settings.prefix}_"
            f"{i:03d}"

        )


        per_candidate = QEInputSettings(
            **settings_dict
        )


        calc_dir = write_relax_input(

            atoms,

            outdir=
                outdir,

            settings=
                per_candidate,

            pseudos=
                pseudos,

        )


        created.append(
            calc_dir
        )


    return created


# ============================================================
# LOAD STRUCTURES FROM DIRECTORY
# ============================================================

def load_structures_from_directory(
    directory: str | Path,
) -> list[Atoms]:
    """
    Load every structure file ASE can understand from a
    directory.
    """

    directory = Path(
        directory
    )


    files = sorted(

        path

        for path in directory.iterdir()

        if path.is_file()

    )


    atoms_list: list[Atoms] = []


    for path in files:

        try:

            atoms_list.append(

                read(
                    str(path)
                )

            )

        except Exception:

            # Ignore files ASE cannot interpret
            # as atomic structures.

            continue


    return atoms_list


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    """
    Command-line interface.

    Keeping argparse inside main() ensures importing this
    module does not attempt to parse command-line arguments.
    """

    parser = argparse.ArgumentParser(

        description=(
            "Prepare Quantum ESPRESSO input "
            "calculations from structures."
        )

    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    parser.add_argument(

        "input",

        type=str,

        help=(
            "Input structure file or directory."
        ),

    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(

        "--outdir",

        type=str,

        default="qe_runs",

        help="Output directory.",

    )


    # --------------------------------------------------------
    # PREFIX
    # --------------------------------------------------------

    parser.add_argument(

        "--prefix",

        type=str,

        default="na_antiperovskite",

        help="QE prefix.",

    )


    # --------------------------------------------------------
    # CALCULATION TYPE
    # --------------------------------------------------------

    parser.add_argument(

        "--calculation",

        type=str,

        default="relax",

        choices=[

            "scf",
            "relax",
            "vc-relax",
            "md",

        ],

        help=(
            "QE calculation type."
        ),

    )


    # --------------------------------------------------------
    # CUTOFFS
    # --------------------------------------------------------

    parser.add_argument(

        "--ecutwfc",

        type=float,

        default=50.0,

    )


    parser.add_argument(

        "--ecutrho",

        type=float,

        default=400.0,

    )


    # --------------------------------------------------------
    # K-POINTS
    # --------------------------------------------------------

    parser.add_argument(

        "--kpts",

        type=int,

        nargs=3,

        default=[
            2,
            2,
            2,
        ],

        metavar=(
            "KX",
            "KY",
            "KZ",
        ),

    )


    # --------------------------------------------------------
    # PSEUDOPOTENTIAL DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(

        "--pseudo-dir",

        type=str,

        default=PSEUDO_DIR,

    )


    # --------------------------------------------------------
    # SYMMETRY
    # --------------------------------------------------------

    parser.add_argument(

        "--use-symmetry",

        action="store_true",

    )


    # --------------------------------------------------------
    # SPIN
    # --------------------------------------------------------

    parser.add_argument(

        "--nspin",

        type=int,

        default=1,

    )


    # --------------------------------------------------------
    # ELECTRONIC CONVERGENCE
    # --------------------------------------------------------

    parser.add_argument(

        "--conv-thr",

        type=float,

        default=1.0e-8,

    )


    # --------------------------------------------------------
    # MD PARAMETERS
    # --------------------------------------------------------

    parser.add_argument(

        "--nstep",

        type=int,

        default=None,

        help=(
            "Number of MD steps."
        ),

    )


    parser.add_argument(

        "--dt",

        type=float,

        default=None,

        help=(
            "QE ionic timestep in atomic units."
        ),

    )


    # --------------------------------------------------------
    # PARSE ARGUMENTS
    # --------------------------------------------------------

    args = parser.parse_args()


    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    settings = QEInputSettings(

        calculation=
            args.calculation,

        prefix=
            args.prefix,

        pseudo_dir=
            args.pseudo_dir,

        ecutwfc=
            args.ecutwfc,

        ecutrho=
            args.ecutrho,

        kpts=
            tuple(
                args.kpts
            ),

        use_symmetry=
            args.use_symmetry,

        nspin=
            args.nspin,

        conv_thr=
            args.conv_thr,

        nstep=
            args.nstep,

        dt=
            args.dt,

    )


    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    input_path = Path(
        args.input
    )


    if input_path.is_dir():

        structures = (
            load_structures_from_directory(
                input_path
            )
        )

    else:

        structures = [

            read(
                str(input_path)
            )

        ]


    # --------------------------------------------------------
    # CREATE CALCULATIONS
    # --------------------------------------------------------

    created = prepare_relaxations(

        structures,

        outdir=
            args.outdir,

        settings=
            settings,

    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    for path in created:

        print(
            path
        )


# ============================================================
# IMPORTANT IMPORT GUARD
# ============================================================

if __name__ == "__main__":

    main()

#!/usr/bin/env python3
"""Create one schema-compliant QMC HDF5 file from a run folder."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.2114
BOHR_TO_ANG = 0.529177


def _force_col_key(name: str) -> Tuple[int, int]:
    match = re.match(r"force_(\d+)_(\d+)$", name)
    if not match:
        return (10**9, 10**9)
    return (int(match.group(1)), int(match.group(2)))


def _parse_extxyz_header(line: str) -> Dict[str, object]:
    pairs = re.findall(r'([A-Za-z0-9_\-]+)=(".*?"|\S+)', line)
    out: Dict[str, object] = {}
    for key, raw in pairs:
        val = raw.strip('"')
        if re.fullmatch(r"[\-+]?\d+", val):
            out[key] = int(val)
        elif re.fullmatch(r"[\-+]?\d*\.?\d+(?:[eE][\-+]?\d+)?", val):
            out[key] = float(val)
        else:
            out[key] = val
    return out


def _formula_from_symbols(symbols: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[0])
    return "".join([f"{el}{n if n > 1 else ''}" for el, n in ordered])


def _read_extxyz(xyz_path: Path) -> Dict[str, object]:
    lines = xyz_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid xyz format in {xyz_path}")
    natoms = int(lines[0].strip())
    header = _parse_extxyz_header(lines[1])
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ atom count mismatch in {xyz_path}: expected {natoms}, found {len(atom_lines)}")

    symbols: List[str] = []
    positions = np.zeros((natoms, 3), dtype=float)
    forces = np.zeros((natoms, 3), dtype=float)
    has_forces = True
    for i, line in enumerate(atom_lines):
        toks = line.split()
        if len(toks) < 4:
            raise ValueError(f"Malformed atom line {i+3} in {xyz_path}")
        symbols.append(toks[0])
        positions[i] = [float(toks[1]), float(toks[2]), float(toks[3])]
        if len(toks) >= 7:
            forces[i] = [float(toks[4]), float(toks[5]), float(toks[6])]
        else:
            has_forces = False

    lattice_raw = str(header.get("Lattice", "")).split()
    if len(lattice_raw) == 9:
        lattice = np.array([float(x) for x in lattice_raw], dtype=float).reshape(3, 3)
    else:
        lattice = np.eye(3)
    frac = positions @ np.linalg.inv(lattice)

    pbc_raw = str(header.get("pbc", "T T T")).split()
    pbc = np.array([x.upper().startswith("T") for x in pbc_raw[:3]], dtype=np.bool_)
    if len(pbc) != 3:
        pbc = np.array([True, True, True], dtype=np.bool_)

    return {
        "natoms": natoms,
        "header": header,
        "symbols": symbols,
        "positions": positions,
        "forces": forces if has_forces else None,
        "lattice": lattice,
        "fractional_positions": frac,
        "pbc": pbc,
        "formula": _formula_from_symbols(symbols),
    }


def extract_twist_averaged_scalar(
    scalar_h5: Path, discard_blocks: int = 32
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract twist-averaged means/errors for each calculation state."""
    data = pd.read_hdf(scalar_h5)
    numeric_cols = data.columns.difference(["calculation", "twist_number"])

    grouped = data.groupby(["calculation", "twist_number"])
    means = grouped[numeric_cols].apply(lambda g: g.iloc[discard_blocks:].mean())
    std_errs = grouped[numeric_cols].apply(
        lambda g: g.iloc[discard_blocks:].apply(
            lambda c: (np.std(c.values, ddof=1) / np.sqrt(len(c))) if len(c) > 1 else np.nan
        )
    )

    weights = pd.Series(1.0, index=means.index)
    wsum = weights.groupby(level="calculation").sum()

    w_means = means.mul(weights, axis=0).groupby(level="calculation").sum().div(wsum, axis=0)
    w_errs = (
        np.sqrt(std_errs.pow(2).mul(weights.pow(2), axis=0).groupby(level="calculation").sum())
        .div(wsum, axis=0)
    )
    return w_means, w_errs


def extract_twist_averaged_dsk(dsk_h5: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read twist-resolved dsk and return twist-averaged mean/error/kvecs."""
    with h5py.File(dsk_h5, "r") as h5f:
        twist_names = sorted([k for k in h5f.keys() if k.startswith("twist")])
        means = []
        errs = []
        kvecs = None
        for tname in twist_names:
            grp = h5f[tname]
            if "dsk_mean" not in grp or "dsk_error" not in grp or "kvecs" not in grp:
                continue
            means.append(np.array(grp["dsk_mean"]))
            errs.append(np.array(grp["dsk_error"]))
            if kvecs is None:
                kvecs = np.array(grp["kvecs"])

    if not means:
        raise ValueError(f"No twist dsk datasets found in {dsk_h5}")
    means_arr = np.vstack(means)
    errs_arr = np.vstack(errs)
    ntwist = means_arr.shape[0]
    mean = means_arr.mean(axis=0)
    err = np.sqrt(np.sum(errs_arr**2, axis=0)) / ntwist
    if kvecs is None:
        raise ValueError(f"No k-vectors found in {dsk_h5}")
    return mean, err, kvecs


def parse_fsc_out(fsc_out: Path) -> Dict[str, float]:
    """Parse finite-size correction terms from fsc.out (Hartree per particle -> eV)."""
    text = fsc_out.read_text(encoding="utf-8", errors="ignore")
    dv = [
        float(x)
        for x in re.findall(
            r"Potential energy correction[\s\S]*?dV/N=\s*([\-+0-9.Ee]+)\s+Ha", text
        )
    ]
    dt = [
        float(x)
        for x in re.findall(
            r"Kinetic energy correction[\s\S]*?dT/N=\s*([\-+0-9.Ee]+)\s+Ha", text
        )
    ]
    out: Dict[str, float] = {}
    if dv:
        out["fsc_potential_energy"] = dv[-1] * HARTREE_TO_EV
    if dt:
        out["fsc_kinetic_energy"] = dt[-1] * HARTREE_TO_EV
    return out


def _decode_species(symbols: Sequence[str]) -> List[str]:
    return sorted(list(set(symbols)))


def build_hdf5_for_run(run_folder: Path, out_h5: Path) -> None:
    xyz_files = sorted(run_folder.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No xyz file found in {run_folder}")
    xyz_path = xyz_files[0]
    xyz_data = _read_extxyz(xyz_path)
    info = dict(xyz_data["header"])

    scalar_files = sorted(run_folder.glob("qmc_scalardata_*.h5"))
    if not scalar_files:
        raise FileNotFoundError(f"No qmc_scalardata_*.h5 found in {run_folder}")
    scalar_h5 = scalar_files[0]

    dsk_vmc = sorted(run_folder.glob("*.s000.dsk.h5"))
    dsk_dmc = sorted(run_folder.glob("*.s001.dsk.h5"))
    if not dsk_vmc or not dsk_dmc:
        raise FileNotFoundError(f"Expected both *.s000.dsk.h5 and *.s001.dsk.h5 in {run_folder}")

    fsc_out = run_folder / "fsc.out"
    fsc_terms = parse_fsc_out(fsc_out) if fsc_out.exists() else {}

    means, errs = extract_twist_averaged_scalar(scalar_h5)
    states = [s for s in ["VMC", "DMC"] if s in means.index]
    if not states:
        raise ValueError(f"Did not find VMC/DMC rows in {scalar_h5}")

    force_cols = sorted([c for c in means.columns if c.startswith("force_")], key=_force_col_key)
    natoms = int(xyz_data["natoms"])
    if len(force_cols) != natoms * 3:
        raise ValueError(
            f"Force columns mismatch: found {len(force_cols)}, expected {natoms*3} "
            f"(natoms={natoms})"
        )

    dsk_vmc_mean, dsk_vmc_err, kvecs = extract_twist_averaged_dsk(dsk_vmc[0])
    dsk_dmc_mean, dsk_dmc_err, _ = extract_twist_averaged_dsk(dsk_dmc[0])

    with h5py.File(out_h5, "w") as h5f:
        # Root attrs
        h5f.attrs["system"] = run_folder.name
        h5f.attrs["formula"] = str(xyz_data["formula"])
        h5f.attrs["natoms"] = natoms
        h5f.attrs["species"] = np.array(_decode_species(xyz_data["symbols"]), dtype="S")
        h5f.attrs["calculation_type"] = "qmc"
        h5f.attrs["method"] = "VMC,DMC"
        h5f.attrs["method_kws"] = np.array(
            ["twist_averaged", "mixed_estimator_available", "qmcpack"], dtype="S"
        )
        h5f.attrs["temperature"] = float(info.get("temperature", np.nan))
        h5f.attrs["pressure"] = float(info.get("pressure", np.nan))
        h5f.attrs["creation_date"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["uuid"] = str(info.get("uuid", ""))

        # /parameters
        grp_par = h5f.create_group("parameters")
        grp_par.create_dataset("kpoint_grid", data=np.array([6, 6, 6], dtype=np.int32))
        grp_par.attrs["spin_polarized"] = False
        grp_par.create_dataset(
            "other_input", data=np.bytes_(json.dumps({"source_xyz_info": info}, default=str))
        )

        # /code
        grp_code = h5f.create_group("code")
        grp_code.attrs["name"] = "QMCPACK"
        grp_code.attrs["version"] = "unknown"
        grp_code.attrs["compiler"] = "unknown"
        grp_code.attrs["parallelization"] = "MPI/OpenMP (unknown split)"
        input_candidates = sorted((run_folder / "dmc").glob("*.s001.qmc.xml"))
        input_text = input_candidates[0].read_text(encoding="utf-8", errors="ignore") if input_candidates else ""
        grp_code.create_dataset("input_file", data=np.bytes_(input_text))
        grp_code.create_dataset("stdout", data=np.bytes_(""))
        grp_code.create_dataset("system", data=np.bytes_(str(info.get("qmc_machine", "Aurora"))))
        grp_code.create_dataset("location", data=np.bytes_(str(run_folder)))

        # /structure
        grp_struct = h5f.create_group("structure")
        grp_struct.create_dataset("lattice_vectors", data=np.array(xyz_data["lattice"]))
        grp_struct.create_dataset("positions", data=np.array(xyz_data["positions"]))
        grp_struct.create_dataset(
            "fractional_positions", data=np.array(xyz_data["fractional_positions"])
        )
        grp_struct.create_dataset("pbc", data=np.array(xyz_data["pbc"], dtype=np.bool_))
        grp_sym = grp_struct.create_group("symmetry")
        grp_sym.create_dataset("spacegroup", data=np.bytes_("unknown"))
        grp_struct.create_dataset(
            "xyz",
            data=np.bytes_(xyz_path.read_text(encoding="utf-8", errors="ignore")),
        )

        # /observables
        grp_obs = h5f.create_group("observables")
        grp_obs.create_dataset("state_labels", data=np.array(states, dtype="S"))

        total_e = np.array([means.loc[s, "LocalEnergy"] for s in states]) * HARTREE_TO_EV
        total_e_err = np.array([errs.loc[s, "LocalEnergy"] for s in states]) * HARTREE_TO_EV
        kin_e = np.array([means.loc[s, "Kinetic"] for s in states]) * HARTREE_TO_EV
        kin_e_err = np.array([errs.loc[s, "Kinetic"] for s in states]) * HARTREE_TO_EV
        pot_e = np.array([means.loc[s, "LocalPotential"] for s in states]) * HARTREE_TO_EV
        pot_e_err = np.array([errs.loc[s, "LocalPotential"] for s in states]) * HARTREE_TO_EV

        grp_obs.create_dataset("total_energy", data=total_e)
        grp_obs.create_dataset("total_energy_error", data=total_e_err)
        grp_obs.create_dataset("kinetic_energy", data=kin_e)
        grp_obs.create_dataset("kinetic_energy_error", data=kin_e_err)
        grp_obs.create_dataset("potential_energy", data=pot_e)
        grp_obs.create_dataset("potential_energy_error", data=pot_e_err)

        force_scale = HARTREE_TO_EV / BOHR_TO_ANG
        force_vals = []
        force_err_vals = []
        for s in states:
            f = means.loc[s, force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
            fe = errs.loc[s, force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
            force_vals.append(f)
            force_err_vals.append(fe)
        grp_obs.create_dataset("forces", data=np.array(force_vals))
        grp_obs.create_dataset("forces_error", data=np.array(force_err_vals))

        sf = np.vstack([dsk_vmc_mean, dsk_dmc_mean])
        sf_err = np.vstack([dsk_vmc_err, dsk_dmc_err])
        grp_obs.create_dataset("structure_factor", data=sf)
        grp_obs.create_dataset("structure_factor_error", data=sf_err)
        grp_obs.create_dataset("structure_factor_ks", data=kvecs)

        if "fsc_potential_energy" in fsc_terms:
            grp_obs.create_dataset("fsc_potential_energy", data=np.array([fsc_terms["fsc_potential_energy"]]))
            grp_obs.create_dataset("fsc_potential_energy_error", data=np.array([np.nan]))
        if "fsc_kinetic_energy" in fsc_terms:
            grp_obs.create_dataset("fsc_kinetic_energy", data=np.array([fsc_terms["fsc_kinetic_energy"]]))
            grp_obs.create_dataset("fsc_kinetic_energy_error", data=np.array([np.nan]))

        # /provenance
        grp_prov = h5f.create_group("provenance")
        grp_prov.create_dataset("dependent_uuids", data=np.array([], dtype="S"))
        grp_prov.create_dataset(
            "source_files",
            data=np.array(
                [
                    str(xyz_path),
                    str(scalar_h5),
                    str(dsk_vmc[0]),
                    str(dsk_dmc[0]),
                    str(fsc_out),
                ],
                dtype="S",
            ),
        )
        grp_prov.create_dataset(
            "notes",
            data=np.bytes_(
                "FSC *_error set to NaN because fsc.out does not report direct uncertainty values."
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one schema-compliant QMC HDF5 file.")
    parser.add_argument(
        "--run-folder",
        type=Path,
        default=Path(
            "/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/run_2025_05_25/runs/LLPT_261_configs/P140T2400config81"
        ),
        help="Path to one configuration folder containing xyz, qmc_scalardata, dsk, fsc files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/database_work/configurations/example_new_schema_P140T2400config81.h5"
        ),
        help="Output HDF5 path.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_hdf5_for_run(args.run_folder, args.output)
    print(f"Wrote schema HDF5: {args.output}")


if __name__ == "__main__":
    main()

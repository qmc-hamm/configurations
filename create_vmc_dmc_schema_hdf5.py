#!/usr/bin/env python3
"""Create separate VMC and DMC HDF5 files using the requested schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    natoms = int(lines[0].strip())
    header = _parse_extxyz_header(lines[1])
    atom_lines = lines[2 : 2 + natoms]

    symbols: List[str] = []
    positions = np.zeros((natoms, 3), dtype=float)
    for i, line in enumerate(atom_lines):
        toks = line.split()
        symbols.append(toks[0])
        positions[i] = [float(toks[1]), float(toks[2]), float(toks[3])]

    lattice_raw = str(header.get("Lattice", "")).split()
    lattice = np.array([float(x) for x in lattice_raw], dtype=float).reshape(3, 3)
    frac = positions @ np.linalg.inv(lattice)
    pbc = np.array(
        [x.upper().startswith("T") for x in str(header.get("pbc", "T T T")).split()[:3]],
        dtype=np.bool_,
    )
    return {
        "natoms": natoms,
        "header": header,
        "symbols": symbols,
        "positions": positions,
        "lattice": lattice,
        "fractional_positions": frac,
        "pbc": pbc if len(pbc) == 3 else np.array([True, True, True], dtype=np.bool_),
        "formula": _formula_from_symbols(symbols),
        "xyz_text": xyz_path.read_text(encoding="utf-8", errors="ignore"),
    }


def extract_twist_averaged_scalar(
    scalar_h5: Path, discard_blocks: int = 32
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_hdf(scalar_h5)
    numeric_cols = data.columns.difference(["calculation", "twist_number"])
    grouped = data.groupby(["calculation", "twist_number"])
    means = grouped[numeric_cols].apply(lambda g: g.iloc[discard_blocks:].mean())
    errs = grouped[numeric_cols].apply(
        lambda g: g.iloc[discard_blocks:].apply(
            lambda c: (np.std(c.values, ddof=1) / np.sqrt(len(c))) if len(c) > 1 else np.nan
        )
    )
    weights = pd.Series(1.0, index=means.index)
    wsum = weights.groupby(level="calculation").sum()
    w_means = means.mul(weights, axis=0).groupby(level="calculation").sum().div(wsum, axis=0)
    w_errs = np.sqrt(errs.pow(2).mul(weights.pow(2), axis=0).groupby(level="calculation").sum()).div(
        wsum, axis=0
    )
    return w_means, w_errs


def extract_twist_averaged_dsk(dsk_h5: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(dsk_h5, "r") as h5f:
        twist_names = sorted([k for k in h5f.keys() if k.startswith("twist")])
        means = []
        errs = []
        kvecs = None
        for tname in twist_names:
            grp = h5f[tname]
            means.append(np.array(grp["dsk_mean"]))
            errs.append(np.array(grp["dsk_error"]))
            if kvecs is None:
                kvecs = np.array(grp["kvecs"])
    means_arr = np.vstack(means)
    errs_arr = np.vstack(errs)
    ntwist = means_arr.shape[0]
    mean = means_arr.mean(axis=0)
    err = np.sqrt(np.sum(errs_arr**2, axis=0)) / ntwist
    return mean, err, kvecs


def parse_fsc_out(fsc_out: Path) -> Dict[str, float]:
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


def _read_nexus_versions(nexus_out: Path) -> Dict[str, str]:
    text = nexus_out.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, str] = {}
    m = re.search(r"Nexus\s+([0-9]+\.[0-9]+\.[0-9]+)", text)
    out["nexus_version"] = m.group(1) if m else "unknown"
    for name in ["python3", "numpy", "scipy", "h5py", "spglib"]:
        mm = re.search(rf"\b{name}\s*=\s*([0-9A-Za-z.\-]+)", text)
        if mm:
            out[f"nexus_dep_{name}"] = mm.group(1)
    return out


def _parse_qmc_params(twist_xml: Path) -> Dict[str, Dict[str, float]]:
    text = twist_xml.read_text(encoding="utf-8", errors="ignore")

    def _block(method: str) -> str:
        m = re.search(rf'<qmc method="{method}"[\s\S]*?</qmc>', text)
        return m.group(0) if m else ""

    def _param(block: str, key: str) -> float:
        m = re.search(rf'<parameter name="{key}"\s*>\s*([\-+0-9.Ee]+)\s*</parameter>', block)
        return float(m.group(1)) if m else np.nan

    vmc = _block("vmc")
    dmc = _block("dmc")
    return {
        "vmc": {
            "time_step": _param(vmc, "timestep"),
            "n_steps": _param(vmc, "steps"),
            "blocks": _param(vmc, "blocks"),
        },
        "dmc": {
            "time_step": _param(dmc, "timestep"),
            "n_steps": _param(dmc, "steps"),
            "blocks": _param(dmc, "blocks"),
        },
    }


def _unique_hex_uuid(old_uuid: str, method: str) -> str:
    return hashlib.sha256(f"{old_uuid}:{method}".encode("utf-8")).hexdigest()


def _read_text_or_empty(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _safe_dataset_name(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    return str(rel).replace("/", "__").replace(".", "_")


def _write_starting_config_generation(
    grp_parent: h5py.Group,
    model_name: str,
    source_base: Path,
    pressure_gpa: int,
    start_method: str,
    simulation_type: str,
    author: str,
    header_info: Dict[str, object],
) -> None:
    grp = grp_parent.create_group("starting_configuration_generation")
    grp.attrs["code"] = "LAMMPS"
    grp.attrs["machine_learned_potential_name"] = model_name
    grp.attrs["source_base"] = str(source_base)
    grp.attrs["method"] = start_method
    grp.attrs["simulation_type"] = simulation_type
    grp.attrs["author"] = author
    if "timestep" in header_info:
        grp.attrs["timestep"] = float(header_info["timestep"])
    if "phase" in header_info:
        grp.attrs["phase"] = str(header_info["phase"])
    if "molecular_percentage" in header_info:
        grp.attrs["molecular_percentage"] = float(header_info["molecular_percentage"])
    if "rs" in header_info:
        grp.attrs["rs"] = float(header_info["rs"])
    if "config_gen_date" in header_info:
        grp.attrs["config_generation_date"] = str(header_info["config_gen_date"])

    # Keep only requested LAMMPS inputs.
    in_mace_files = sorted(source_base.glob("in.mace.*"))
    for fp in in_mace_files:
        dname = _safe_dataset_name(fp, source_base)
        grp.create_dataset(dname, data=np.bytes_(_read_text_or_empty(fp)))

    pressure_data = source_base / f"p{pressure_gpa}" / "data.txt"
    if pressure_data.exists():
        grp.create_dataset("data_txt", data=np.bytes_(_read_text_or_empty(pressure_data)))


def _write_method_file(
    out_h5: Path,
    method: str,
    run_folder: Path,
    xyz_data: Dict[str, object],
    means: pd.DataFrame,
    errs: pd.DataFrame,
    dsk_mean: np.ndarray,
    dsk_err: np.ndarray,
    kvecs: np.ndarray,
    fsc_terms: Dict[str, float],
    force_cols: List[str],
    old_uuid: str,
    run_script: Path,
    nexus_out: Path,
    qsub_file: Path,
    qmc_xml: Path,
    qmc_params: Dict[str, Dict[str, float]],
    starting_cfg_src: Path,
) -> None:
    method_upper = method.upper()
    method_key = method.lower()
    info = dict(xyz_data["header"])
    natoms = int(xyz_data["natoms"])

    raw_total = float(means.loc[method_upper, "LocalEnergy"]) * HARTREE_TO_EV
    raw_total_err = float(errs.loc[method_upper, "LocalEnergy"]) * HARTREE_TO_EV
    raw_kin = float(means.loc[method_upper, "Kinetic"]) * HARTREE_TO_EV
    raw_kin_err = float(errs.loc[method_upper, "Kinetic"]) * HARTREE_TO_EV
    raw_pot = float(means.loc[method_upper, "LocalPotential"]) * HARTREE_TO_EV
    raw_pot_err = float(errs.loc[method_upper, "LocalPotential"]) * HARTREE_TO_EV

    # DMC includes FSC correction per request.
    if method_upper == "DMC":
        total_e = raw_total + fsc_terms.get("fsc_potential_energy", 0.0) + fsc_terms.get(
            "fsc_kinetic_energy", 0.0
        )
        kin_e = raw_kin + fsc_terms.get("fsc_kinetic_energy", 0.0)
        pot_e = raw_pot + fsc_terms.get("fsc_potential_energy", 0.0)
    else:
        total_e = raw_total
        kin_e = raw_kin
        pot_e = raw_pot

    force_scale = HARTREE_TO_EV / BOHR_TO_ANG
    if method_upper == "DMC":
        # Mixed estimator: 2*DMC - VMC
        dmc_f = means.loc["DMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        vmc_f = means.loc["VMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        dmc_fe = errs.loc["DMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        vmc_fe = errs.loc["VMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        forces = 2.0 * dmc_f - vmc_f
        forces_err = np.sqrt(4.0 * dmc_fe**2 + vmc_fe**2)
        mixed_est = True
    else:
        forces = means.loc["VMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        forces_err = errs.loc["VMC", force_cols].to_numpy(dtype=float).reshape(natoms, 3) * force_scale
        mixed_est = False

    nexus_versions = _read_nexus_versions(nexus_out)
    creation_date = str(info.get("config_gen_date", ""))
    if not creation_date:
        u = str(info.get("uuid", ""))
        m = re.search(r"(\d{4}-\d{2}-\d{2})$", u)
        creation_date = m.group(1) if m else str(info.get("QMC-run-date", ""))

    config_number = int(info.get("config", -1))
    model_name = str(info.get("modelname", "unknown"))
    author = str(info.get("author", "unknown"))
    pressure_gpa = int(float(info.get("pressure", np.nan)))
    start_method = str(info.get("method", "unknown"))
    simulation_type = str(info.get("simulation-type", "unknown"))
    phase = str(info.get("phase", "unknown"))

    with h5py.File(out_h5, "w") as h5f:
        # Root attrs (indexable metadata)
        h5f.attrs["system"] = run_folder.name
        h5f.attrs["formula"] = str(xyz_data["formula"])
        h5f.attrs["natoms"] = natoms
        h5f.attrs["species"] = np.array(_decode_species(xyz_data["symbols"]), dtype="S")
        h5f.attrs["calculation_type"] = "qmc"
        h5f.attrs["method"] = method_upper
        h5f.attrs["method_kws"] = np.array(
            ["twist_averaged", "qmcpack_complex", "legacy_driver"], dtype="S"
        )
        h5f.attrs["temperature"] = float(info.get("temperature", np.nan))
        h5f.attrs["pressure"] = float(info.get("pressure", np.nan))
        h5f.attrs["creation_date"] = creation_date
        h5f.attrs["uuid"] = _unique_hex_uuid(old_uuid, method_upper)
        h5f.attrs["original_uuid"] = old_uuid
        h5f.attrs["config_number"] = config_number
        h5f.attrs["mixed_estimator_available"] = mixed_est
        h5f.attrs["starting_configuration_model_name"] = model_name
        h5f.attrs["qmcpack_version"] = "4.0-cpu-complex"
        h5f.attrs["nexus_version"] = nexus_versions.get("nexus_version", "unknown")
        h5f.attrs["author"] = author
        if "rs" in info:
            h5f.attrs["rs"] = float(info["rs"])
        if "QMC-quality" in info:
            h5f.attrs["qmc_quality"] = int(info["QMC-quality"])
        if "QMC-run-date" in info:
            h5f.attrs["qmc_run_date"] = str(info["QMC-run-date"])
        for k, v in nexus_versions.items():
            if k != "nexus_version":
                h5f.attrs[k] = v

        # /parameters
        grp_par = h5f.create_group("parameters")
        grp_par.create_dataset("kpoint_grid", data=np.array([6, 6, 6], dtype=np.int32))
        grp_par.attrs["spin_polarized"] = False
        grp_par.create_dataset("time_step", data=np.array([qmc_params[method_key]["time_step"]]))
        grp_par.create_dataset("n_steps", data=np.array([qmc_params[method_key]["n_steps"]]))
        grp_par.create_dataset("other_input", data=np.bytes_(json.dumps({"xyz_header": info}, default=str)))

        # /code
        grp_code = h5f.create_group("code")
        grp_code.attrs["name"] = "QMCPACK"
        grp_code.attrs["version"] = "4.0-cpu-complex"
        grp_code.attrs["compiler"] = "oneapi/eng-compiler/2024.07.30.002"
        grp_code.attrs["parallelization"] = "MPI+OpenMP"
        grp_code.create_dataset("input_file", data=np.bytes_(_read_text_or_empty(run_script)))
        grp_code.create_dataset("stdout", data=np.bytes_(_read_text_or_empty(nexus_out)))
        grp_code.create_dataset("system", data=np.bytes_(str(info.get("qmc_machine", "Aurora"))))
        grp_code.create_dataset("method_qsub", data=np.bytes_(_read_text_or_empty(qsub_file)))
        grp_code.create_dataset("method_twist_xml", data=np.bytes_(_read_text_or_empty(qmc_xml)))

        _write_starting_config_generation(
            grp_parent=grp_code,
            model_name=model_name,
            source_base=starting_cfg_src,
            pressure_gpa=pressure_gpa,
            start_method=start_method,
            simulation_type=simulation_type,
            author=author,
            header_info=info,
        )

        # /structure
        grp_struct = h5f.create_group("structure")
        grp_struct.create_dataset("lattice_vectors", data=np.array(xyz_data["lattice"]))
        grp_struct.create_dataset("positions", data=np.array(xyz_data["positions"]))
        grp_struct.create_dataset(
            "fractional_positions", data=np.array(xyz_data["fractional_positions"])
        )
        grp_struct.create_dataset("pbc", data=np.array(xyz_data["pbc"], dtype=np.bool_))
        grp_struct.create_dataset("xyz", data=np.bytes_(str(xyz_data["xyz_text"])))

        # /observables (single method file)
        grp_obs = h5f.create_group("observables")
        grp_obs.create_dataset("total_energy", data=np.array([total_e]))
        grp_obs.create_dataset("total_energy_error", data=np.array([raw_total_err]))
        grp_obs.create_dataset("kinetic_energy", data=np.array([kin_e]))
        grp_obs.create_dataset("kinetic_energy_error", data=np.array([raw_kin_err]))
        grp_obs.create_dataset("potential_energy", data=np.array([pot_e]))
        grp_obs.create_dataset("potential_energy_error", data=np.array([raw_pot_err]))
        grp_obs.create_dataset("forces", data=forces[None, :, :])
        grp_obs.create_dataset("forces_error", data=forces_err[None, :, :])
        grp_obs.create_dataset("structure_factor", data=dsk_mean)
        grp_obs.create_dataset("structure_factor_error", data=dsk_err)
        grp_obs.create_dataset("structure_factor_ks", data=kvecs)

        if method_upper == "DMC":
            # Use corrected energies from extxyz header when available.
            if "energy" in info:
                grp_obs["total_energy"][...] = np.array([float(info["energy"])])
            if "electron_kinetic_energy" in info:
                grp_obs["kinetic_energy"][...] = np.array([float(info["electron_kinetic_energy"])])
            if "potential_energy" in info:
                grp_obs["potential_energy"][...] = np.array([float(info["potential_energy"])])
            if "fsc_potential_energy" in fsc_terms:
                grp_obs.create_dataset(
                    "fsc_potential_energy", data=np.array([fsc_terms["fsc_potential_energy"]])
                )
            if "fsc_kinetic_energy" in fsc_terms:
                grp_obs.create_dataset(
                    "fsc_kinetic_energy", data=np.array([fsc_terms["fsc_kinetic_energy"]])
                )
            grp_obs.create_dataset("total_energy_uncorrected", data=np.array([raw_total]))
            grp_obs.create_dataset("kinetic_energy_uncorrected", data=np.array([raw_kin]))
            grp_obs.create_dataset("potential_energy_uncorrected", data=np.array([raw_pot]))

        # /provenance
        grp_prov = h5f.create_group("provenance")
        if method_upper == "DMC":
            vmc_uuid = _unique_hex_uuid(old_uuid, "VMC")
            grp_prov.create_dataset("dependent_uuids", data=np.array([vmc_uuid], dtype="S"))
        else:
            grp_prov.create_dataset("dependent_uuids", data=np.array([], dtype="S"))
        grp_prov.create_dataset(
            "source_files",
            data=np.array(
                [str(run_folder), str(run_script), str(nexus_out), str(qsub_file), str(qmc_xml)],
                dtype="S",
            ),
        )


def build_vmc_dmc_files(run_folder: Path, output_prefix: Path) -> Tuple[Path, Path]:
    xyz_files = sorted(run_folder.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No xyz file found in {run_folder}")
    xyz_path = xyz_files[0]
    xyz_data = _read_extxyz(xyz_path)
    info = dict(xyz_data["header"])
    old_uuid = str(info.get("uuid", ""))

    scalar_h5 = sorted(run_folder.glob("qmc_scalardata_*.h5"))[0]
    means, errs = extract_twist_averaged_scalar(scalar_h5)

    dsk_vmc = sorted(run_folder.glob("*.s000.dsk.h5"))[0]
    dsk_dmc = sorted(run_folder.glob("*.s001.dsk.h5"))[0]
    vmc_sf, vmc_sf_err, vmc_k = extract_twist_averaged_dsk(dsk_vmc)
    dmc_sf, dmc_sf_err, dmc_k = extract_twist_averaged_dsk(dsk_dmc)

    fsc_out = run_folder / "fsc.out"
    fsc_terms = parse_fsc_out(fsc_out) if fsc_out.exists() else {}

    force_cols = sorted([c for c in means.columns if c.startswith("force_")], key=_force_col_key)
    natoms = int(xyz_data["natoms"])
    if len(force_cols) != natoms * 3:
        raise ValueError(f"Found {len(force_cols)} force cols but expected {natoms*3}.")

    run_root = run_folder.parents[2]  # run_YYYY_MM_DD
    run_script = run_root / "run_QMC_chiesa_force.py"
    nexus_out = run_root / f"{run_folder.parents[0].name}.out"
    if not nexus_out.exists():
        outs = sorted(run_root.glob("*.out"))
        nexus_out = outs[0] if outs else run_root / "nexus.out"

    qsub_vmc = run_folder / "opt" / "pbe-opt.qsub.in"
    qsub_dmc = run_folder / "dmc" / "pbe-dmc.qsub.in"
    qmc_xml = run_folder / "dmc" / "pbe-dmc.g000.twistnum_0.in.xml"
    qmc_params = _parse_qmc_params(qmc_xml)

    starting_cfg_src = Path("/projects/illinois/grants/qmchamm/shared/shubhang/MACE_w_LAMMPS/M18")

    vmc_out = output_prefix.parent / f"{output_prefix.name}_VMC.h5"
    dmc_out = output_prefix.parent / f"{output_prefix.name}_DMC.h5"

    _write_method_file(
        out_h5=vmc_out,
        method="VMC",
        run_folder=run_folder,
        xyz_data=xyz_data,
        means=means,
        errs=errs,
        dsk_mean=vmc_sf,
        dsk_err=vmc_sf_err,
        kvecs=vmc_k,
        fsc_terms=fsc_terms,
        force_cols=force_cols,
        old_uuid=old_uuid,
        run_script=run_script,
        nexus_out=nexus_out,
        qsub_file=qsub_vmc,
        qmc_xml=qmc_xml,
        qmc_params=qmc_params,
        starting_cfg_src=starting_cfg_src,
    )

    _write_method_file(
        out_h5=dmc_out,
        method="DMC",
        run_folder=run_folder,
        xyz_data=xyz_data,
        means=means,
        errs=errs,
        dsk_mean=dmc_sf,
        dsk_err=dmc_sf_err,
        kvecs=dmc_k,
        fsc_terms=fsc_terms,
        force_cols=force_cols,
        old_uuid=old_uuid,
        run_script=run_script,
        nexus_out=nexus_out,
        qsub_file=qsub_dmc,
        qmc_xml=qmc_xml,
        qmc_params=qmc_params,
        starting_cfg_src=starting_cfg_src,
    )

    return vmc_out, dmc_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create separate VMC and DMC schema HDF5 files.")
    parser.add_argument(
        "--run-folder",
        type=Path,
        default=Path(
            "/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/run_2025_05_25/runs/LLPT_261_configs/P140T2400config81"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "/projects/illinois/grants/qmchamm/shared/shubhang/aurora_backup/database_work/configurations/P140T2400config81_schema"
        ),
    )
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    vmc_out, dmc_out = build_vmc_dmc_files(args.run_folder, args.output_prefix)
    print(f"Wrote: {vmc_out}")
    print(f"Wrote: {dmc_out}")


if __name__ == "__main__":
    main()

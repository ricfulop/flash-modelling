"""Helpers for the Sunday Tier A morphology/fraction sweep."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from . import percolation as perc


REPO_ROOT = Path(__file__).resolve().parents[2]
TIER_A_DIR = REPO_ROOT / "data" / "tier_a"


def frac_slug(frac: float) -> str:
    """Stable directory slug for a defect fraction."""
    return f"{float(frac):.2f}".replace(".", "p")


def iter_cases(cfg: dict):
    """Yield (morphology, fraction) cases from the frozen experiment config."""
    exp = cfg.get("experiment", {})
    morphologies = exp.get("morphologies", [cfg["network"]["morphology"]])
    fractions = exp.get("frac_sweep", [cfg["network"]["target_frac"]])
    for morphology in morphologies:
        for frac in fractions:
            yield str(morphology), float(frac)


def case_dir(morphology: str, frac: float, root: Path = TIER_A_DIR) -> Path:
    return Path(root) / morphology / f"frac_{frac_slug(frac)}"


def build_network(cfg: dict, morphology: str, frac: float):
    """Build one morphology/fraction hypothesis network."""
    n = cfg["network"]
    nx_, ny, nz = n["supercell"]
    kwargs = dict(nx_=nx_, ny=ny, nz=nz, a=cfg["lattice_a"], seed=n.get("seed", 0))
    if morphology == "ordered":
        return perc.ordered_superlattice(frac=frac, **kwargs)
    if morphology == "random":
        return perc.random_percolation(frac=frac, **kwargs)
    if morphology in {"dendrite", "dendritic"}:
        return perc.dla_dendrite(target_frac=frac, **kwargs)
    raise ValueError(f"unknown Tier A morphology: {morphology}")


def save_network(net, out_dir: Path, metadata: dict, spectral_t_max: int = 4000,
                 spectral_walkers: int = 4000):
    """Write network arrays, metadata, and geometry report for one case."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "network.npz",
        coords=net.coords,
        mask=net.defect_mask,
        a=np.array(net.a),
        nx_shape=np.array(net.nx_shape, dtype=int),
    )
    geometry = perc.classify(
        net,
        spectral_t_max=spectral_t_max,
        spectral_walkers=spectral_walkers,
        seed=int(metadata.get("seed", 0)),
    )
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "geometry.json", geometry)
    return geometry


def load_network(path: Path):
    """Load a saved Tier A network from a case directory or network.npz path."""
    path = Path(path)
    npz_path = path if path.name.endswith(".npz") else path / "network.npz"
    data = np.load(npz_path)
    return perc.from_arrays(
        data["coords"],
        data["mask"],
        float(np.asarray(data["a"])),
        tuple(int(x) for x in data.get("nx_shape", np.array([0, 0, 0]))),
    )


def to_builtin(value):
    """Convert numpy/dataclass values into JSON-serializable Python objects."""
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())

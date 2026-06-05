#!/usr/bin/env python3
"""Diagnose blocked CHGNet NEB image-continuity QC without changing results."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from electrodefect import mlip_al  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def load_band_images(summary: dict) -> list:
    return [read(path) for path in summary["image_paths"]]


def nearest_atom_to_targets(images, targets: np.ndarray) -> list[dict]:
    rows = []
    for image_index, (image, target) in enumerate(zip(images, targets, strict=True)):
        positions = image.get_positions()
        distances = np.linalg.norm(positions - target[None, :], axis=1)
        nearest = int(np.argmin(distances))
        rows.append(
            {
                "image": image_index,
                "nearest_atom_index": nearest,
                "nearest_distance_A": float(distances[nearest]),
            }
        )
    return rows


def greedy_nearest_track(images, start_index: int) -> list[dict]:
    rows = []
    current_index = int(start_index)
    rows.append({"image": 0, "tracked_atom_index": current_index, "step_distance_A": 0.0})
    for image_index in range(1, len(images)):
        prev_position = images[image_index - 1].get_positions()[current_index]
        positions = images[image_index].get_positions()
        distances = np.linalg.norm(positions - prev_position[None, :], axis=1)
        current_index = int(np.argmin(distances))
        rows.append(
            {
                "image": image_index,
                "tracked_atom_index": current_index,
                "step_distance_A": float(distances[current_index]),
            }
        )
    return rows


def direct_index_displacements(images, atom_index: int) -> list[float]:
    positions = np.array([image.get_positions()[atom_index] for image in images])
    return np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(float).tolist()


def largest_direct_index_jumps(images, top_n: int = 5) -> list[dict]:
    rows = []
    for pair_index in range(len(images) - 1):
        a = images[pair_index].get_positions()
        b = images[pair_index + 1].get_positions()
        distances = np.linalg.norm(b - a, axis=1)
        order = np.argsort(distances)[::-1][:top_n]
        rows.append(
            {
                "pair": [pair_index, pair_index + 1],
                "largest_jumps": [
                    {"atom_index": int(idx), "distance_A": float(distances[idx])}
                    for idx in order
                ],
            }
        )
    return rows


def linear_targets(initial, final, atom_index: int, n_images: int) -> np.ndarray:
    start = initial.get_positions()[atom_index]
    stop = final.get_positions()[atom_index]
    weights = np.linspace(0.0, 1.0, n_images)
    return np.array([(1.0 - weight) * start + weight * stop for weight in weights])


def diagnose_band(name: str, run_root: Path, manifest: dict) -> dict:
    summary = read_json(run_root / name / "neb_summary.json")
    images = load_band_images(summary)
    mobile = int(manifest["mobile_atom_index"])
    primary = manifest.get("primary", {})
    adjacent_limit = float(primary.get("max_adjacent_mobile_displacement_A", 1.0))
    targets = linear_targets(images[0], images[-1], mobile, len(images))
    target_nearest = nearest_atom_to_targets(images, targets)
    greedy_track = greedy_nearest_track(images, mobile)
    direct_steps = direct_index_displacements(images, mobile)
    min_pairs = [mlip_al.min_pair_distance(image) for image in images]

    target_indices = [row["nearest_atom_index"] for row in target_nearest]
    greedy_indices = [row["tracked_atom_index"] for row in greedy_track]
    same_mobile_to_targets = all(idx == mobile for idx in target_indices)
    greedy_preserves_identity = all(idx == mobile for idx in greedy_indices)
    direct_jump_max = float(max(direct_steps)) if direct_steps else 0.0

    if not same_mobile_to_targets:
        likely_cause = "atom_role_exchange_or_endpoint_path_instability"
    elif not greedy_preserves_identity:
        likely_cause = "nearest_neighbor_identity_switch"
    elif direct_jump_max > adjacent_limit:
        likely_cause = "same_atom_large_relaxation_jump"
    else:
        likely_cause = "no_image_continuity_anomaly_detected"

    return {
        "band": name,
        "mobile_atom_index": mobile,
        "barrier_eV": summary["barrier_eV"],
        "image_qc_passes": summary["image_qc"]["passes"],
        "direct_mobile_adjacent_displacements_A": direct_steps,
        "direct_mobile_max_adjacent_displacement_A": direct_jump_max,
        "adjacent_mobile_limit_A": adjacent_limit,
        "nearest_atom_to_linear_mobile_target": target_nearest,
        "greedy_nearest_track_from_mobile": greedy_track,
        "target_indices_all_mobile": same_mobile_to_targets,
        "greedy_track_preserves_mobile_identity": greedy_preserves_identity,
        "min_pair_distances_A": min_pairs,
        "largest_direct_index_jumps": largest_direct_index_jumps(images),
        "likely_cause": likely_cause,
    }


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# CHGNet NEB blocked-run diagnostic",
        "",
        f"Run root: `{payload['run_root']}`",
        "",
        "## Summary",
        "",
        f"- Overall diagnosis: `{payload['overall_diagnosis']}`",
        f"- Interpretation: {payload['interpretation']}",
        "",
        "## Band diagnostics",
        "",
    ]
    for band in payload["bands"]:
        lines.extend(
            [
                f"### `{band['band']}`",
                "",
                f"- Barrier: `{band['barrier_eV']:.6f} eV`",
                f"- Image QC passes: `{band['image_qc_passes']}`",
                f"- Max selected-mobile adjacent jump: "
                f"`{band['direct_mobile_max_adjacent_displacement_A']:.6f} A`",
                f"- Adjacent selected-mobile limit: `{band['adjacent_mobile_limit_A']:.6f} A`",
                f"- Target-nearest indices all selected mobile: "
                f"`{band['target_indices_all_mobile']}`",
                f"- Greedy nearest track preserves selected identity: "
                f"`{band['greedy_track_preserves_mobile_identity']}`",
                f"- Likely band cause: `{band['likely_cause']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Next action",
            "",
            "Do not promote this run to BigDFT. If continuing, write a fresh revised "
            "endpoint/path preregistration that either constrains the intended atom path, "
            "uses a more local endpoint, or explicitly treats atom exchange as the pathway.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def diagnose(run_root: Path) -> dict:
    manifest = read_json(run_root / "endpoint_manifest.json")
    bands = [diagnose_band("no_field", run_root, manifest), diagnose_band("field_primary", run_root, manifest)]
    causes = {band["likely_cause"] for band in bands}
    if "atom_role_exchange_or_endpoint_path_instability" in causes:
        overall = "endpoint_path_instability_or_atom_role_exchange"
        interpretation = (
            "The selected mobile atom is not consistently the atom nearest the intended "
            "linear path after NEB relaxation. The QC failure is therefore likely a real "
            "path-definition issue, not just a scalar-barrier reporting issue."
        )
    elif "nearest_neighbor_identity_switch" in causes:
        overall = "atom_identity_tracking_ambiguous"
        interpretation = (
            "A nearest-neighbor track switches atom identity, so the saved band should be "
            "reviewed as a possible exchange path before changing QC."
        )
    elif "same_atom_large_relaxation_jump" in causes:
        overall = "same_atom_large_relaxation_jump"
        interpretation = (
            "The selected atom remains the path-nearest atom but moves too far between "
            "adjacent images, suggesting too few images or an under-resolved endpoint."
        )
    else:
        overall = "no_continuity_root_cause_detected"
        interpretation = "The scripted diagnostics did not reproduce the continuity anomaly."
    payload = {
        "run_root": run_root,
        "overall_diagnosis": overall,
        "interpretation": interpretation,
        "bands": bands,
    }
    write_json(run_root / "neb_continuity_diagnostic.json", payload)
    write_report(run_root / "neb_continuity_diagnostic.md", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    payload = diagnose(args.run_root)
    print(json.dumps(jsonable({"overall_diagnosis": payload["overall_diagnosis"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

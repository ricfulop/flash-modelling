#!/usr/bin/env python3
"""Build an interactive HTML gallery for ASE-readable structure outputs.

The gallery is intentionally dependency-light: ASE reads the structures and the
HTML uses 3Dmol.js in the browser. It is useful for quick inspection of defect
slabs, NEB images, DFT input/output geometries, and close-contact failures.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import Bohr


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "runs"
DEFAULT_OUT = REPO / "runs" / "ase_structure_gallery_latest"
SUPPORTED_SUFFIXES = {".xyz", ".extxyz", ".cif", ".traj"}


@dataclass
class StructureCard:
    id: str
    source: str
    source_rel: str
    name: str
    formula: str
    natoms: int
    pbc: list[bool]
    cell_lengths: list[float]
    min_distance_a: float | None
    min_pair: list[int] | None
    modified: str
    xyz: str


def iter_structure_paths(root: Path, suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(root.rglob(f"*{suffix}"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def min_distance(atoms) -> tuple[float | None, list[int] | None]:
    n = len(atoms)
    if n < 2 or n > 2500:
        return None, None
    try:
        distances = atoms.get_all_distances(mic=bool(any(atoms.pbc)))
    except Exception:
        distances = atoms.get_all_distances(mic=False)
    distances[distances <= 1.0e-12] = np.inf
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    value = float(distances[idx])
    if not math.isfinite(value):
        return None, None
    return value, [int(idx[0]), int(idx[1])]


def atoms_to_xyz(atoms) -> str:
    stream = StringIO()
    write(stream, atoms, format="xyz")
    return stream.getvalue()


def read_bigdft_posinp(path: Path) -> Atoms:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("not enough lines for BigDFT posinp format")

    header = lines[0].split()
    natoms = int(header[0])
    unit = header[1].lower() if len(header) > 1 else "angstroem"
    scale = Bohr if unit.startswith("bohr") else 1.0

    boundary = lines[1].split()
    bc = boundary[0].lower()
    cell_values = [float(x) * scale for x in boundary[1:4]]
    if len(cell_values) != 3:
        raise ValueError("BigDFT posinp cell line does not contain three lengths")
    if bc == "periodic":
        pbc = [True, True, True]
    elif bc == "surface":
        pbc = [True, True, False]
    elif bc == "wire":
        pbc = [False, False, True]
    else:
        pbc = [False, False, False]

    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid atom line: {line!r}")
        symbols.append(fields[0])
        positions.append([float(fields[1]) * scale, float(fields[2]) * scale, float(fields[3]) * scale])

    return Atoms(symbols=symbols, positions=positions, cell=cell_values, pbc=pbc)


def read_atoms(path: Path) -> Atoms:
    try:
        return read(path, index=-1)
    except Exception as ase_error:
        try:
            return read_bigdft_posinp(path)
        except Exception as bigdft_error:
            raise RuntimeError(f"ASE: {ase_error}; BigDFT posinp: {bigdft_error}") from bigdft_error


def load_card(path: Path, idx: int) -> StructureCard | None:
    try:
        atoms = read_atoms(path)
    except Exception as exc:
        print(f"skip {path}: {exc}")
        return None
    dmin, pair = min_distance(atoms)
    lengths = atoms.cell.lengths() if atoms.cell is not None else [0.0, 0.0, 0.0]
    stat = path.stat()
    return StructureCard(
        id=f"s{idx:04d}",
        source=str(path),
        source_rel=safe_rel(path),
        name=path.parent.name + "/" + path.name,
        formula=atoms.get_chemical_formula(),
        natoms=len(atoms),
        pbc=[bool(x) for x in atoms.pbc],
        cell_lengths=[round(float(x), 6) for x in lengths],
        min_distance_a=round(dmin, 6) if dmin is not None else None,
        min_pair=pair,
        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        xyz=atoms_to_xyz(atoms),
    )


def render_html(cards: list[StructureCard], generated_at: str) -> str:
    payload = json.dumps([asdict(card) for card in cards])
    rows = "\n".join(
        f"""
        <button class="structure-button" data-id="{card.id}">
          <span class="button-title">{html.escape(card.name)}</span>
          <span class="button-meta">{html.escape(card.formula)} · {card.natoms} atoms · dmin {card.min_distance_a if card.min_distance_a is not None else 'n/a'} Å</span>
        </button>
        """
        for card in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ASE Structure Gallery</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101217;
      color: #eef1f7;
    }}
    body {{ margin: 0; }}
    header {{
      padding: 18px 22px;
      border-bottom: 1px solid #2b3140;
      background: #151923;
    }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .subtle {{ color: #9ba7bd; font-size: 13px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      height: calc(100vh - 78px);
    }}
    aside {{
      overflow: auto;
      border-right: 1px solid #2b3140;
      padding: 12px;
      background: #11151d;
    }}
    .toolbar {{ display: flex; gap: 8px; margin-bottom: 12px; }}
    input {{
      flex: 1;
      padding: 9px 10px;
      border-radius: 8px;
      border: 1px solid #30384a;
      background: #0d1017;
      color: #eef1f7;
    }}
    .structure-button {{
      width: 100%;
      display: block;
      text-align: left;
      padding: 10px 11px;
      margin: 0 0 8px;
      border: 1px solid #293044;
      border-radius: 10px;
      color: #eef1f7;
      background: #171c27;
      cursor: pointer;
    }}
    .structure-button:hover, .structure-button.active {{
      border-color: #79a6ff;
      background: #1c2638;
    }}
    .button-title {{ display: block; font-size: 13px; line-height: 1.35; }}
    .button-meta {{ display: block; color: #9ba7bd; margin-top: 5px; font-size: 12px; }}
    section {{ display: grid; grid-template-rows: auto minmax(0, 1fr); min-width: 0; }}
    #details {{
      padding: 14px 18px;
      border-bottom: 1px solid #2b3140;
      background: #11151d;
    }}
    #details h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .kv {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{
      border: 1px solid #30384a;
      background: #171c27;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      color: #cbd3e3;
    }}
    .warning {{ color: #ffd27d; border-color: #8f6a20; }}
    #viewer {{ width: 100%; height: 100%; min-height: 480px; background: #07090d; }}
    code {{ color: #c7dcff; }}
  </style>
</head>
<body>
  <header>
    <h1>ASE Structure Gallery</h1>
    <div class="subtle">Generated {html.escape(generated_at)} · {len(cards)} structures · newest files first</div>
  </header>
  <main>
    <aside>
      <div class="toolbar"><input id="search" placeholder="Filter by path, formula, run..." /></div>
      <div id="list">{rows}</div>
    </aside>
    <section>
      <div id="details"><h2>Select a structure</h2><div class="subtle">Use the list to inspect geometry, cell, and close contacts.</div></div>
      <div id="viewer"></div>
    </section>
  </main>
  <script>
    const structures = {payload};
    const byId = new Map(structures.map(s => [s.id, s]));
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#07090d" }});
    const details = document.getElementById("details");

    function render(id) {{
      const s = byId.get(id);
      if (!s) return;
      document.querySelectorAll(".structure-button").forEach(b => b.classList.toggle("active", b.dataset.id === id));
      viewer.clear();
      viewer.addModel(s.xyz, "xyz");
      viewer.setStyle({{}}, {{ sphere: {{ scale: 0.32 }}, stick: {{ radius: 0.16 }} }});
      viewer.addUnitCell();
      viewer.zoomTo();
      viewer.render();
      const warn = s.min_distance_a !== null && s.min_distance_a < 1.8;
      details.innerHTML = `
        <h2>${{escapeHtml(s.name)}}</h2>
        <div class="kv">
          <span class="chip">${{escapeHtml(s.formula)}}</span>
          <span class="chip">${{s.natoms}} atoms</span>
          <span class="chip">pbc ${{s.pbc.join(", ")}}</span>
          <span class="chip">cell ${{s.cell_lengths.join(" × ")}} Å</span>
          <span class="chip ${{warn ? "warning" : ""}}">min distance ${{s.min_distance_a ?? "n/a"}} Å ${{s.min_pair ? "(" + s.min_pair.join(", ") + ")" : ""}}</span>
          <span class="chip">modified ${{escapeHtml(s.modified)}}</span>
        </div>
        <p class="subtle"><code>${{escapeHtml(s.source_rel)}}</code></p>
      `;
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    document.getElementById("list").addEventListener("click", (event) => {{
      const button = event.target.closest(".structure-button");
      if (button) render(button.dataset.id);
    }});

    document.getElementById("search").addEventListener("input", (event) => {{
      const q = event.target.value.toLowerCase();
      document.querySelectorAll(".structure-button").forEach(button => {{
        const s = byId.get(button.dataset.id);
        const haystack = `${{s.source_rel}} ${{s.formula}} ${{s.name}}`.toLowerCase();
        button.style.display = haystack.includes(q) ? "" : "none";
      }});
    }});

    if (structures.length) render(structures[0].id);
  </script>
</body>
</html>
"""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory to scan recursively.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Directory for index.html and manifest.")
    p.add_argument("--max-files", type=int, default=160, help="Maximum newest structure files to include.")
    p.add_argument("--suffix", action="append", default=None, help="File suffix to include; may repeat.")
    return p


def main() -> int:
    args = parser().parse_args()
    suffixes = {s if s.startswith(".") else f".{s}" for s in args.suffix} if args.suffix else SUPPORTED_SUFFIXES
    paths = iter_structure_paths(args.root, suffixes)[: max(1, int(args.max_files))]
    cards: list[StructureCard] = []
    for path in paths:
        card = load_card(path, len(cards))
        if card is not None:
            cards.append(card)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    (args.output_dir / "index.html").write_text(render_html(cards, generated_at))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "root": str(args.root),
                "n_structures": len(cards),
                "structures": [{k: v for k, v in asdict(card).items() if k != "xyz"} for card in cards],
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
percolation.py — Defect-network geometry for the electrodefect simulation.

Builds a percolating / dendritic network of Frenkel-pair (vacancy + SIA) sites
on a BCC tungsten lattice, and computes the geometric diagnostics that classify
the transport regime:

    - spanning check + site-percolation threshold (BCC p_c ~ 0.246)
    - fractal (Hausdorff) dimension D_f via box counting
    - spectral (fracton) dimension d_s via random-walk return probability
        p(t) ~ t^(-d_s/2)            [Alexander-Orbach: d_s ~ 4/3 at criticality]
    - walk dimension d_w = 2 D_f / d_s

These are the goal-1 discriminators. d_s ~ 4/3 => Alexander-Orbach universality
(fracton/quantum-percolation regime, NOT Anderson). d_s -> 3 => approaching a
dense extended (miniband) network.

Pure numpy + networkx. No external simulation engine needed. Tested standalone.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from dataclasses import dataclass, field


# BCC nearest-neighbour offsets (8 of them), in units of the conventional cell a.
BCC_NN = np.array([
    [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5],
    [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5, -0.5, -0.5],
])
BCC_PC_SITE = 0.246          # site percolation threshold, BCC (literature)


@dataclass
class DefectNetwork:
    """A defect network on a BCC supercell."""
    coords: np.ndarray                      # (N,3) all BCC site coords (cartesian, Angstrom)
    defect_mask: np.ndarray                 # (N,) bool, True = defect (Frenkel-pair) site
    a: float                                # lattice parameter (Angstrom)
    nx_shape: tuple                         # (nx, ny, nz) conventional cells
    graph: nx.Graph = field(default=None)   # connectivity graph of defect sites

    @property
    def defect_indices(self) -> np.ndarray:
        return np.flatnonzero(self.defect_mask)

    @property
    def frac(self) -> float:
        return float(self.defect_mask.mean())


def bcc_sites(nx_: int, ny: int, nz: int, a: float) -> np.ndarray:
    """All BCC lattice sites (corner + body-centre) of an nx*ny*nz supercell."""
    corners, bodies = [], []
    for i in range(nx_):
        for j in range(ny):
            for k in range(nz):
                corners.append([i, j, k])
                bodies.append([i + 0.5, j + 0.5, k + 0.5])
    sites = np.array(corners + bodies, dtype=float) * a
    return sites


def _nn_graph(coords: np.ndarray, a: float, tol: float = 0.05) -> nx.Graph:
    """Nearest-neighbour graph over the given coords (BCC NN distance = sqrt(3)/2 a)."""
    from scipy.spatial import cKDTree
    nn_dist = np.sqrt(3) / 2 * a
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=nn_dist * (1 + tol))
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    G.add_edges_from(pairs)
    return G


def random_percolation(nx_=12, ny=12, nz=12, a=3.165, frac=0.25,
                       seed=0) -> DefectNetwork:
    """
    Standard random site percolation: occupy a fraction `frac` of BCC sites as
    Frenkel-pair defects, then keep only the largest connected (spanning) cluster.
    Use when you want canonical percolation statistics.
    """
    rng = np.random.default_rng(seed)
    coords = bcc_sites(nx_, ny, nz, a)
    N = len(coords)
    occupied = rng.random(N) < frac
    G_all = _nn_graph(coords[occupied], a)
    # map back to global indices
    occ_idx = np.flatnonzero(occupied)
    if G_all.number_of_nodes() == 0:
        mask = np.zeros(N, bool)
        return DefectNetwork(coords, mask, a, (nx_, ny, nz), nx.Graph())
    comps = sorted(nx.connected_components(G_all), key=len, reverse=True)
    largest_local = comps[0]
    largest_global = occ_idx[list(largest_local)]
    mask = np.zeros(N, bool)
    mask[largest_global] = True
    net = DefectNetwork(coords, mask, a, (nx_, ny, nz))
    net.graph = _nn_graph(coords[mask], a)
    return net


def ordered_superlattice(nx_=16, ny=16, nz=16, a=3.165, frac=0.25,
                         seed=0) -> DefectNetwork:
    """
    Deterministic ordered defect phase used as the miniband hypothesis.

    The model occupies a centered set of adjacent half-lattice x-layers. This
    keeps the defect network crystalline and connected at 25-35 mol% while
    preserving the same BCC nearest-neighbour graph contract as the random and
    dendritic hypotheses.
    """
    coords = bcc_sites(nx_, ny, nz, a)
    half_x = np.rint(coords[:, 0] / a * 2).astype(int)
    n_layers = half_x.max() + 1
    target_layers = int(np.clip(round(frac * n_layers), 2, n_layers))
    offset = int(seed) % n_layers
    center = n_layers // 2
    layers = np.unique(half_x)
    rel = (layers - center - offset + n_layers) % n_layers
    rel = np.minimum(rel, n_layers - rel)
    chosen_layers = layers[np.argsort(rel)[:target_layers]]
    allowed = set(chosen_layers.tolist())
    mask = np.array([x in allowed for x in half_x], dtype=bool)
    net = DefectNetwork(coords, mask, a, (nx_, ny, nz))
    net.graph = _nn_graph(coords[mask], a)
    return net


def dla_dendrite(nx_=16, ny=16, nz=16, a=3.165, target_frac=0.25,
                 n_seeds=1, stick=1.0, seed=0) -> DefectNetwork:
    """
    Diffusion-limited-aggregation dendrite: grow a connected, fractal, *dendritic*
    defect cluster by random-walk accretion onto seed(s). This is the morphology
    that matches a drive-grown defect network (branched, percolating backbone with
    dead-end side branches). Returns when the cluster reaches `target_frac`.

    `stick` < 1 thins/branches the dendrite (lower D_f); = 1 is greedy DLA.
    """
    rng = np.random.default_rng(seed)
    coords = bcc_sites(nx_, ny, nz, a)
    N = len(coords)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    nn_dist = np.sqrt(3) / 2 * a

    in_cluster = np.zeros(N, bool)
    # seed near the centre (this becomes the surface-facing backbone root if you
    # orient the supercell so +z is the emission face; see build.py)
    centre = coords.mean(0)
    for _ in range(n_seeds):
        s = np.argmin(np.linalg.norm(coords - centre, axis=1))
        in_cluster[s] = True

    target = int(target_frac * N)
    n_in = int(in_cluster.sum())
    box_min, box_max = coords.min(0), coords.max(0)
    max_tries = 200 * target
    tries = 0
    while n_in < target and tries < max_tries:
        tries += 1
        # launch a walker at a random site, random-walk on BCC NN until it
        # touches the cluster, then stick (with prob `stick`).
        w = rng.integers(N)
        for _step in range(4 * (nx_ + ny + nz)):
            neigh = tree.query_ball_point(coords[w], nn_dist * 1.05)
            neigh = [m for m in neigh if m != w]
            if not neigh:
                break
            if any(in_cluster[m] for m in neigh):
                if rng.random() < stick:
                    in_cluster[w] = True
                    n_in += 1
                break
            w = neigh[rng.integers(len(neigh))]
    net = DefectNetwork(coords, in_cluster, a, (nx_, ny, nz))
    net.graph = _nn_graph(coords[in_cluster], a)
    return net


def from_arrays(coords: np.ndarray, defect_mask: np.ndarray, a: float,
                nx_shape=None) -> DefectNetwork:
    """Reconstruct a DefectNetwork from saved arrays."""
    coords = np.asarray(coords, float)
    defect_mask = np.asarray(defect_mask, bool)
    if nx_shape is None:
        nx_shape = (0, 0, 0)
    net = DefectNetwork(coords, defect_mask, float(a), tuple(nx_shape))
    net.graph = _nn_graph(coords[defect_mask], float(a))
    return net


# --------------------------------------------------------------------------- #
# Geometric diagnostics
# --------------------------------------------------------------------------- #
def is_spanning(net: DefectNetwork, axis: int = 2) -> bool:
    """Does the defect cluster span the cell along `axis` (0=x,1=y,2=z)?"""
    if net.graph.number_of_nodes() == 0:
        return False
    pts = net.coords[net.defect_indices]
    lo, hi = pts[:, axis].min(), pts[:, axis].max()
    span = net.coords[:, axis].max() - net.coords[:, axis].min()
    return bool((hi - lo) > 0.9 * span)


def fractal_dimension(net: DefectNetwork, n_scales: int = 8) -> float:
    """Box-counting Hausdorff dimension D_f of the defect cluster."""
    pts = net.coords[net.defect_indices]
    if len(pts) < 8:
        return float("nan")
    mins, maxs = pts.min(0), pts.max(0)
    L = (maxs - mins).max()
    sizes = L / np.logspace(0.3, np.log10(len(pts) ** (1 / 3) + 1), n_scales)
    counts = []
    for eps in sizes:
        keys = set(map(tuple, np.floor((pts - mins) / eps).astype(int)))
        counts.append(len(keys))
    counts = np.array(counts, float)
    good = counts > 0
    slope = np.polyfit(np.log(1 / sizes[good]), np.log(counts[good]), 1)[0]
    return float(slope)


def spectral_dimension(net: DefectNetwork, t_max: int = 4000,
                       n_walkers: int = 4000, seed: int = 0):
    """
    Spectral (fracton) dimension d_s from the random-walk return probability on
    the defect graph:  P_return(t) ~ t^(-d_s/2).

    Returns (d_s, t_grid, P_return). Alexander-Orbach: d_s ~ 4/3 at percolation.
    """
    G = net.graph
    if G.number_of_edges() == 0:
        return float("nan"), None, None
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}
    nodes_arr = np.array(nodes)
    starts = rng.choice(nodes_arr, size=min(n_walkers, len(nodes_arr) * 4))
    ts = np.unique(np.logspace(0, np.log10(t_max), 30).astype(int))
    ts = ts[ts >= 1]
    returns = np.zeros(len(ts), float)
    for s in starts:
        pos = s
        ret_times = set()
        check = set(ts.tolist())
        for t in range(1, ts.max() + 1):
            nb = adj[pos]
            if not nb:
                break
            pos = nb[rng.integers(len(nb))]
            if pos == s and t in check:
                ret_times.add(t)
        for i, t in enumerate(ts):
            if t in ret_times:
                returns[i] += 1
    P = returns / len(starts)
    good = P > 0
    if good.sum() < 4:
        return float("nan"), ts, P
    slope = np.polyfit(np.log(ts[good]), np.log(P[good]), 1)[0]
    d_s = -2 * slope
    return float(d_s), ts, P


def classify(net: DefectNetwork, spectral_t_max: int = 4000,
             spectral_walkers: int = 4000, seed: int = 0) -> dict:
    """One-call geometric classification of the defect network."""
    D_f = fractal_dimension(net)
    d_s, _, _ = spectral_dimension(
        net, t_max=spectral_t_max, n_walkers=spectral_walkers, seed=seed
    )
    d_w = 2 * D_f / d_s if (d_s and not np.isnan(d_s) and d_s > 0) else float("nan")
    regime = "unknown"
    if not np.isnan(d_s):
        if d_s < 1.6:
            regime = "Alexander-Orbach / quantum-percolation (fracton, NOT Anderson)"
        elif d_s > 2.6:
            regime = "dense extended network (approaching miniband)"
        else:
            regime = "intermediate critical network"
    return dict(frac=net.frac, n_defects=int(net.defect_mask.sum()),
                spanning=is_spanning(net), p_c_bcc=BCC_PC_SITE,
                D_f=D_f, d_s=d_s, d_w=d_w, regime=regime,
                spectral_t_max=int(spectral_t_max),
                spectral_walkers=int(spectral_walkers))


if __name__ == "__main__":
    # smoke test both morphologies
    for name, net in [("random", random_percolation(seed=1)),
                      ("dendrite", dla_dendrite(seed=1))]:
        print(f"\n=== {name} ===")
        for k, v in classify(net).items():
            print(f"  {k:12s}: {v}")

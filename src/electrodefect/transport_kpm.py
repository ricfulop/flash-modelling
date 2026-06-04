"""
transport_kpm.py — Linear-scaling quantum transport (LSQT) on the defect network.

Runs on the GB10 GPU (torch, FP32/mixed — DFT stays on the Grace CPUs). Builds a
tight-binding Hamiltonian on the percolating defect graph and computes the goal-1
quantum diagnostics that distinguish miniband / fracton / Anderson:

  - kpm_dos              : Chebyshev-KPM density of states (Jackson kernel, stochastic trace)
  - kubo_greenwood_dc    : KPM Kubo-Greenwood DC conductivity sigma(E) [2D Chebyshev]
  - ipr_spectrum         : energy-resolved inverse participation ratio (exact, small N)
  - level_spacing_ratio  : <r> statistic; GOE~0.531 (extended) vs Poisson~0.386 (localized)
  - density_matrix_decay : xi from rho(r,r') ~ exp(-|r-r'|/xi) (localization length)

Scaling: KPM DOS / KG are O(N * M * R) and run to 1e5-1e6 sites on one GB10.
IPR / level stats / rho-decay use exact eigh and are for the smaller DFT-sized cells
(<~ few thousand sites) — use them to *calibrate* the KPM on the same small cell, then
trust KPM on the big disordered cells.

NOTE: written against the standard KPM-Kubo formulation (Weisse RMP 2006;
Garcia-Covaci-Rappoport PRL 2015). NOT executed in this environment (no torch here) —
validate kpm_dos against a small exact-diag DOS on first run (calibration helper provided).
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np

try:
    import torch
    warnings.filterwarnings(
        "ignore",
        message="Sparse invariant checks are implicitly disabled.*",
        category=UserWarning,
    )
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def network_from_atoms(atoms, lattice_a=None, cutoff_A=None, cutoff_scale=1.35):
    """
    Convert an ASE structure into the graph contract used by KPM transport.

    This is a geometry-proxy bridge for Phase 2 plumbing: one orbital is assigned to
    each atom and edges follow near-neighbour distances in the relaxed/annealed geometry.
    It is not a DFT/Wannier downfolding substitute.
    """
    import networkx as nx
    from scipy.spatial import cKDTree
    from .percolation import DefectNetwork

    coords = np.asarray(atoms.get_positions(), dtype=float)
    n_atoms = len(coords)
    mask = np.ones(n_atoms, dtype=bool)
    if n_atoms == 0:
        net = DefectNetwork(coords, mask, float(lattice_a or 1.0), (0, 0, 0))
        net.graph = nx.Graph()
        return net
    if cutoff_A is None:
        if n_atoms < 2:
            cutoff_A = float(lattice_a or 1.0)
        else:
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=min(2, n_atoms))
            nn = distances[:, 1]
            nn = nn[np.isfinite(nn) & (nn > 1.0e-8)]
            if len(nn) == 0:
                cutoff_A = float(lattice_a or 1.0)
            else:
                cutoff_A = float(np.median(nn) * cutoff_scale)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    if n_atoms > 1:
        tree = cKDTree(coords)
        graph.add_edges_from(tree.query_pairs(r=float(cutoff_A)))
    if lattice_a is None:
        cell = getattr(atoms, "cell", None)
        lengths = np.asarray(cell.lengths(), dtype=float) if cell is not None else np.array([])
        positive = lengths[np.isfinite(lengths) & (lengths > 1.0e-8)]
        lattice_a = float(np.min(positive)) if len(positive) else float(cutoff_A)
    net = DefectNetwork(coords, mask, float(lattice_a), (0, 0, 0))
    net.graph = graph
    return net


def load_wannier90_hr(path, home_cell=(0, 0, 0), imag_tol=1.0e-8):
    """
    Load the Gamma/home-cell block from a Wannier90 `*_hr.dat` Hamiltonian.

    Returns a scipy COO matrix in eV. This is the preferred Phase 2 transport
    bridge when Wannier/downfolded artifacts exist.
    """
    from scipy.sparse import coo_matrix

    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Wannier90 hr file is too short: {path}")
    try:
        n_wann = int(lines[1].split()[0])
        n_rpts = int(lines[2].split()[0])
    except Exception as exc:
        raise ValueError(f"could not parse Wannier90 hr header in {path}") from exc
    degeneracies: list[int] = []
    idx = 3
    while len(degeneracies) < n_rpts and idx < len(lines):
        degeneracies.extend(int(float(x)) for x in lines[idx].split())
        idx += 1
    if len(degeneracies) < n_rpts:
        raise ValueError(f"missing degeneracy list in {path}")

    target = tuple(int(x) for x in home_cell)
    rows, cols, vals = [], [], []
    for line in lines[idx:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        rx, ry, rz = (int(parts[0]), int(parts[1]), int(parts[2]))
        if (rx, ry, rz) != target:
            continue
        # Wannier90 indices are 1-based: i, j, Re(H), Im(H)
        i = int(parts[3]) - 1
        j = int(parts[4]) - 1
        value = complex(float(parts[5]), float(parts[6]))
        rows.append(i)
        cols.append(j)
        vals.append(value)
    if not vals:
        raise ValueError(f"no home-cell matrix elements found for R={target} in {path}")
    matrix = coo_matrix((vals, (rows, cols)), shape=(n_wann, n_wann)).tocsr()
    matrix = 0.5 * (matrix + matrix.getH())
    if matrix.dtype.kind == "c":
        imag = np.max(np.abs(matrix.data.imag)) if matrix.nnz else 0.0
        if imag > imag_tol:
            raise ValueError(
                f"complex Wannier Hamiltonian needs complex KPM support; max imaginary={imag:g}"
            )
        matrix = matrix.real
    return matrix.tocoo()


def matrix_to_torch_sparse(matrix, device="cuda", dtype=None):
    """Convert a scipy sparse/dense Hamiltonian to a torch sparse COO tensor."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required for matrix_to_torch_sparse")
    from scipy import sparse

    dtype = dtype or torch.float32
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(np.asarray(matrix))
    else:
        matrix = matrix.tocoo()
    idx = torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long, device=device)
    val = torch.tensor(np.asarray(matrix.data, dtype=np.float32), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, matrix.shape, device=device, check_invariants=False).coalesce()


def downfolded_hamiltonian_scalars(matrix, exact_limit=2000):
    """Summarize onsite/hopping/bandwidth scalars from a downfolded Hamiltonian."""
    from scipy import sparse

    H = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))
    diag = H.diagonal().astype(float)
    off = H.copy()
    off.setdiag(0.0)
    off.eliminate_zeros()
    hop_abs = np.abs(off.data.astype(float)) if off.nnz else np.array([], dtype=float)
    payload = {
        "n_orbitals": int(H.shape[0]),
        "n_hoppings": int(off.nnz),
        "onsite_mean_eV": float(np.mean(diag)) if len(diag) else float("nan"),
        "onsite_std_eV": float(np.std(diag)) if len(diag) else float("nan"),
        "onsite_width_eV": float(np.max(diag) - np.min(diag)) if len(diag) else float("nan"),
        "max_hop_eV": float(np.max(hop_abs)) if len(hop_abs) else 0.0,
        "mean_abs_hop_eV": float(np.mean(hop_abs)) if len(hop_abs) else 0.0,
    }
    if H.shape[0] <= exact_limit:
        eig = np.linalg.eigvalsh(H.toarray())
        payload.update(
            {
                "bandwidth_eV": float(eig.max() - eig.min()) if len(eig) else 0.0,
                "eigen_min_eV": float(eig.min()) if len(eig) else float("nan"),
                "eigen_max_eV": float(eig.max()) if len(eig) else float("nan"),
            }
        )
    else:
        payload["bandwidth_eV"] = None
    return payload


# --------------------------------------------------------------------------- #
# Tight-binding Hamiltonian from the defect graph
# --------------------------------------------------------------------------- #
def build_tb(net, t_hop=0.08, onsite=0.0, disorder_W=0.0, seed=0,
             dtype=None, device="cuda"):
    """
    Tight-binding H on the defect network (one orbital per defect site).
      t_hop      : nearest-neighbour transfer integral (eV) — from DFT (transfer integral
                   between adjacent Frenkel-pair sites; see dft_bigdft / Wannier downfold)
      onsite     : mean on-site energy (eV)
      disorder_W : box disorder width (eV); 0 => ordered network (miniband test),
                   >0 => add Anderson disorder for the random comparison
    Returns (H_sparse, X_diag, idx_map): torch sparse COO H, position-along-z diagonal,
    and the global->local index map.
    """
    import networkx as nx
    G = net.graph
    nodes = list(G.nodes())
    idx_map = {g: i for i, g in enumerate(nodes)}
    N = len(nodes)
    rng = np.random.default_rng(seed)
    onsite_e = onsite + (rng.uniform(-disorder_W / 2, disorder_W / 2, N) if disorder_W else 0.0)
    node_arr = np.array(nodes, dtype=int)
    defect_idx = net.defect_indices
    if len(defect_idx) and node_arr.max(initial=-1) < len(defect_idx):
        coords = net.coords[defect_idx[node_arr]]
    else:
        coords = net.coords[node_arr]
    z = coords[:, 2].astype(np.float64)

    rows, cols, vals = [], [], []
    for i in range(N):
        rows.append(i); cols.append(i); vals.append(float(np.atleast_1d(onsite_e)[i] if disorder_W else onsite))
    for u, v in G.edges():
        a, b = idx_map[u], idx_map[v]
        rows += [a, b]; cols += [b, a]; vals += [-t_hop, -t_hop]

    if not _HAS_TORCH or device == "scipy":
        # scipy fallback for calibration/offline use
        from scipy.sparse import coo_matrix
        H = coo_matrix((vals, (rows, cols)), shape=(N, N))
        return H, z, idx_map

    dtype = dtype or torch.float32
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=dtype, device=device)
    H = torch.sparse_coo_tensor(
        idx, val, (N, N), device=device, check_invariants=False
    ).coalesce()
    X = torch.tensor(z, dtype=dtype, device=device)
    return H, X, idx_map


def _rescale(H, emin, emax, eps=0.05):
    """Rescale H -> Htil with spectrum in [-1+eps, 1-eps] for Chebyshev."""
    a = (emax - emin) / (2 - eps)
    b = (emax + emin) / 2
    return a, b


def estimate_spectral_bounds(H, n_iter=40, device="cuda"):
    """Cheap power-iteration bound on |E|max for the rescaling."""
    N = H.shape[0]
    v = torch.randn(N, device=device)
    v /= v.norm()
    lam = 0.0
    for _ in range(n_iter):
        w = torch.sparse.mm(H, v.unsqueeze(1)).squeeze(1)
        lam = w.norm().item()
        v = w / (lam + 1e-30)
    return -1.05 * lam, 1.05 * lam


def kpm_dos(H, M=2048, R=20, n_energies=2000, device="cuda", seed=0):
    """
    Chebyshev-KPM density of states with Jackson kernel and stochastic trace.
    Returns (E, dos). M = #moments (energy resolution ~ bandwidth/M), R = #random vectors.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch required for kpm_dos; run on the GB10.")
    torch.manual_seed(seed)
    N = H.shape[0]
    emin, emax = estimate_spectral_bounds(H, device=device)
    a, b = _rescale(H, emin, emax)

    # Jackson kernel
    n = torch.arange(M, device=device, dtype=torch.float32)
    g = ((M - n + 1) * torch.cos(np.pi * n / (M + 1)) +
         torch.sin(np.pi * n / (M + 1)) / np.tan(np.pi / (M + 1))) / (M + 1)

    mu = torch.zeros(M, device=device)
    for _ in range(R):
        v0 = torch.randn(N, device=device)
        v0 /= v0.norm()
        # T0, T1
        tjm1 = v0.clone()
        # Htil @ v = (H v - b v)/a
        tj = (torch.sparse.mm(H, v0.unsqueeze(1)).squeeze(1) - b * v0) / a
        mu[0] += (v0 @ tjm1)
        mu[1] += (v0 @ tj)
        for m in range(2, M):
            tjp1 = 2 * (torch.sparse.mm(H, tj.unsqueeze(1)).squeeze(1) - b * tj) / a - tjm1
            mu[m] += (v0 @ tjp1)
            tjm1, tj = tj, tjp1
    mu = mu / (R * N) * g

    # reconstruct DOS on Chebyshev nodes
    x = torch.linspace(-1 + 1e-3, 1 - 1e-3, n_energies, device=device)
    Tk = torch.zeros(M, n_energies, device=device)
    Tk[0] = 1.0
    Tk[1] = x
    for m in range(2, M):
        Tk[m] = 2 * x * Tk[m - 1] - Tk[m - 2]
    factor = 1.0 / (np.pi * torch.sqrt(1 - x ** 2))
    dos = factor * (mu[0] + 2 * torch.sum(mu[1:].unsqueeze(1) * Tk[1:], dim=0))
    E = a * x + b
    return E.cpu().numpy(), (dos / a).cpu().numpy()


def kubo_greenwood_dc(H, X, M=1024, R=20, n_energies=400, device="cuda", seed=0):
    """
    KPM Kubo-Greenwood DC conductivity sigma(E) (arbitrary units; calibrate prefactor
    against a reference). vx = i[H, X], X diagonal position along transport (z) axis.
    Implements the 2D Chebyshev expansion of Tr[vx delta(E-H) vx delta(E-H)].

    Returns (E, sigma_dc). Sweep Te later by integrating sigma(E) against -df/dE.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch required for kubo_greenwood_dc; run on the GB10.")
    torch.manual_seed(seed)
    N = H.shape[0]
    emin, emax = estimate_spectral_bounds(H, device=device)
    a, b = _rescale(H, emin, emax)

    def apply_h(v):
        if torch.is_complex(v):
            real = torch.sparse.mm(H, v.real.unsqueeze(1)).squeeze(1)
            imag = torch.sparse.mm(H, v.imag.unsqueeze(1)).squeeze(1)
            return torch.complex(real, imag)
        return torch.sparse.mm(H, v.unsqueeze(1)).squeeze(1)

    def Htil(v):
        return (apply_h(v) - b * v) / a

    # velocity operator action: vx v = i[H,X] v = i(H(Xv) - X(Hv))
    Xv = X
    def vx(v):
        Hv = apply_h(v)
        HXv = apply_h(Xv * v)
        return 1j * (HXv - Xv * Hv)

    # 2D moment matrix mu[m,n] = (1/N) Tr[ vx T_m(Htil) vx T_n(Htil) ]
    mu = torch.zeros(M, M, dtype=torch.complex64, device=device)
    for _ in range(R):
        r = torch.randn(N, device=device).to(torch.complex64)
        r /= r.norm()
        vr = vx(r)                              # vx |r>
        # build {T_n(Htil) vx |r>}
        Tn = [vr.clone(), Htil(vr).to(torch.complex64)]  # n=0,1 (Htil real-symmetric)
        for n in range(2, M):
            Tn.append(2 * Htil(Tn[-1]).to(torch.complex64) - Tn[-2])
        # <r| vx T_m  — build left vectors similarly from vx^dagger r = -vx r (anti-herm)
        lr = vx(r)
        Lm = [lr.clone(), Htil(lr).to(torch.complex64)]
        for m in range(2, M):
            Lm.append(2 * Htil(Lm[-1]).to(torch.complex64) - Lm[-2])
        for m in range(M):
            for n in range(M):
                mu[m, n] += torch.vdot(Lm[m], Tn[n])
    mu /= (R * N)

    # Jackson kernel both indices
    k = torch.arange(M, device=device, dtype=torch.float32)
    g = ((M - k + 1) * torch.cos(np.pi * k / (M + 1)) +
         torch.sin(np.pi * k / (M + 1)) / np.tan(np.pi / (M + 1))) / (M + 1)
    mu = mu * g.unsqueeze(1) * g.unsqueeze(0)

    x = torch.linspace(-1 + 1e-3, 1 - 1e-3, n_energies, device=device)
    Tk = torch.zeros(M, n_energies, device=device)
    Tk[0] = 1.0; Tk[1] = x
    for m in range(2, M):
        Tk[m] = 2 * x * Tk[m - 1] - Tk[m - 2]
    denom = (np.pi ** 2) * (1 - x ** 2)
    sig = torch.einsum('mn,mE,nE->E', mu.real, Tk, Tk) / denom
    E = a * x + b
    return E.cpu().numpy(), sig.cpu().numpy()


# --------------------------------------------------------------------------- #
# Exact diagnostics (small cells; calibrate KPM here)
# --------------------------------------------------------------------------- #
def ipr_spectrum(H_dense):
    """Energy-resolved IPR. P2(E)=sum|psi|^4. ~1/N extended, ->O(1) localized."""
    import numpy.linalg as la
    H = np.asarray(H_dense)
    w, v = la.eigh(H)
    ipr = np.sum(np.abs(v) ** 4, axis=0)
    return w, ipr


def defect_node_coords(net):
    """Coordinates aligned to net.graph node ordering."""
    nodes = np.array(list(net.graph.nodes()), dtype=int)
    defect_idx = net.defect_indices
    if len(nodes) == 0:
        return np.empty((0, 3), dtype=float)
    if len(defect_idx) and nodes.max(initial=-1) < len(defect_idx):
        return net.coords[defect_idx[nodes]]
    return net.coords[nodes]


def level_spacing_ratio(H_dense):
    """
    <r> statistic of consecutive gap ratios. GOE ~0.531 (extended/chaotic),
    Poisson ~0.386 (localized/integrable).
    """
    import numpy.linalg as la
    w = np.sort(la.eigvalsh(np.asarray(H_dense)))
    s = np.diff(w)
    s = s[s > 1e-12]
    if len(s) < 2:
        return float("nan")
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r))


def density_matrix_decay(H_dense, coords, E_F=0.0, Te_eV=0.1):
    """
    Localization length xi from one-body density matrix decay rho(r,r')~exp(-|r-r'|/xi).
    (BigDFT's linear-scaling kernel measures this natively; this is the TB-level check.)
    Returns (distances, |rho|, xi_fit).
    """
    import numpy.linalg as la
    w, v = la.eigh(np.asarray(H_dense))
    f = 1.0 / (1.0 + np.exp((w - E_F) / Te_eV))
    rho = (v * f) @ v.T.conj()
    N = len(coords)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    dd, rr = d[iu], np.abs(rho[iu])
    # bin and fit exponential tail
    order = np.argsort(dd)
    dd, rr = dd[order], rr[order]
    good = rr > 1e-8
    if good.sum() < 5:
        return dd, rr, float("inf")
    slope = np.polyfit(dd[good], np.log(rr[good]), 1)[0]
    xi = -1.0 / slope if slope < 0 else float("inf")
    return dd, rr, float(xi)


def exact_diagnostics(net, t_hop=0.08, disorder_W=0.0, E_F=0.0,
                      Te_eV=0.72, seed=0, max_exact_sites=None):
    """
    Exact finite-cell diagnostics for the frozen regime decision.

    If max_exact_sites is set and the graph is larger, use a deterministic
    highest-degree calibration subgraph. KPM/DOS can still run on the full graph.
    """
    work_net = net
    if max_exact_sites and net.graph.number_of_nodes() > max_exact_sites:
        import networkx as nx
        chosen = [
            n for n, _ in sorted(
                net.graph.degree, key=lambda item: (-item[1], item[0])
            )[:max_exact_sites]
        ]
        sub = net.graph.subgraph(chosen).copy()
        sub_nodes = list(sub.nodes())
        mapping = {old: i for i, old in enumerate(sub_nodes)}
        sub = nx.relabel_nodes(sub, mapping)
        coords = defect_node_coords(net)[sub_nodes]
        mask = np.ones(len(coords), dtype=bool)
        from .percolation import DefectNetwork
        work_net = DefectNetwork(coords, mask, net.a, net.nx_shape, sub)

    H_sp, _, _ = build_tb(work_net, t_hop=t_hop, disorder_W=disorder_W,
                          seed=seed, device="scipy")
    Hd = H_sp.toarray() if hasattr(H_sp, "toarray") else H_sp.to_dense().cpu().numpy()
    energies, ipr = ipr_spectrum(Hd)
    N = len(energies)
    n_window = max(5, int(0.05 * N)) if N else 0
    if n_window:
        order = np.argsort(np.abs(energies - E_F))[:n_window]
        pr_frac = np.mean(1.0 / np.clip(ipr[order], 1e-300, None)) / N
    else:
        pr_frac = float("nan")
    coords = defect_node_coords(work_net)
    _, _, xi = density_matrix_decay(Hd, coords, E_F=E_F, Te_eV=Te_eV)
    bandwidth = float(energies.max() - energies.min()) if N else 0.0
    mean_degree = float(np.mean([d for _, d in work_net.graph.degree()])) if N else 0.0
    kF_l_proxy = mean_degree * t_hop / max(disorder_W, t_hop)
    return {
        "n_sites_exact": int(N),
        "bandwidth_eV": bandwidth,
        "r_stat": level_spacing_ratio(Hd) if N > 2 else float("nan"),
        "pr_frac_EF": float(pr_frac),
        "xi_over_a": float(xi / work_net.a) if np.isfinite(xi) else float("inf"),
        "kF_l": float(kF_l_proxy),
        "W_over_B": float(disorder_W / bandwidth) if bandwidth else float("inf"),
        "diagnostic_note": (
            "kF_l is a TB connectivity proxy until DFT/Kubo calibration; "
            "frozen regime thresholds are otherwise unchanged."
        ),
    }


def calibrate_kpm_vs_exact(net, t_hop=0.08, disorder_W=0.0, M=512, R=10):
    """Run KPM DOS and exact DOS on the same small cell; report agreement."""
    H_sp, z, _ = build_tb(net, t_hop=t_hop, disorder_W=disorder_W, device="scipy")
    if hasattr(H_sp, "toarray"):
        Hd = H_sp.toarray()
    else:
        Hd = H_sp.to_dense().cpu().numpy()
    import numpy.linalg as la
    ev = la.eigvalsh(Hd)
    print(f"exact band: [{ev.min():.3f}, {ev.max():.3f}] eV, N={len(ev)}")
    print(f"<r> level stat = {level_spacing_ratio(Hd):.3f} "
          f"(GOE~0.531 extended, Poisson~0.386 localized)")
    return ev


if __name__ == "__main__":
    # offline (no torch) sanity: build TB on a small dendrite, exact diagnostics
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from percolation import dla_dendrite
    net = dla_dendrite(nx_=8, ny=8, nz=8, target_frac=0.25, seed=1)
    ev = calibrate_kpm_vs_exact(net, t_hop=0.08, disorder_W=0.0)
    print("ordered network band width:", ev.max() - ev.min(), "eV (miniband B)")
    evd = calibrate_kpm_vs_exact(net, t_hop=0.08, disorder_W=3.0)
    print("with W=3 eV disorder, band width:", evd.max() - evd.min(), "eV")

# Slurm cluster setup (flash-dgx)

Unified two-node Slurm cluster for `flash-modelling`:

| Node | Role | DGX link | Repo path |
|------|------|----------|-----------|
| `spark-5da5` | controller + compute | `192.168.100.1` | `/home/ricfulop/Desktop/Cursor/flash-modelling` |
| `spark-0808` (`dgx-peer`) | compute only | `192.168.100.2` | `/home/nvidia/Cursor/flash-modelling` |

Both machines currently had **separate single-node Slurm clusters**. `spark-0808` was **DOWN** with reason `Node unexpectedly rebooted`. This setup federates them under one controller on `spark-5da5`.

## Install (run on spark-5da5)

```bash
cd /home/ricfulop/Desktop/Cursor/flash-modelling
git pull origin main   # after these files are pushed

sudo env/slurm/install-cluster.sh controller
sudo env/slurm/install-cluster.sh compute
sudo env/slurm/install-cluster.sh status
```

What the installer does:

1. Installs shared `slurm.conf`, prolog/epilog, GRES, and profile wrappers to `/etc/slurm/`
2. Adds `/etc/hosts` entries for the DGX peer link (`192.168.100.1/2`)
3. On `spark-5da5`: runs `slurmctld` + `slurmd`
4. On `spark-0808`: **disables** local `slurmctld`, syncs the Munge key, runs `slurmd` against `spark-5da5`
5. Resumes both nodes (`scontrol update ... State=RESUME`)

If a node stays DOWN after reboot:

```bash
sudo env/slurm/install-cluster.sh resume-nodes
```

## Environment profiles

Profiles live in `env/slurm/profiles/` and are installed to `/etc/slurm/flash-profiles/`:

| Profile | Use for | Activation |
|---------|---------|------------|
| `kpm` | Torch/KPM GPU sweeps | `env/activate_dgx.sh` |
| `bigdft` | BigDFT CPU labels | `use_bigdft.sh` |
| `qe-cpu` | QE 7.4.1 CPU smoke | `use_qe_cpu.sh` |
| `qe-gpu` | QE 7.5 GPU labels | NVHPC + QE 7.5 paths (spark-5da5 only today) |

Run any command under a profile:

```bash
env/slurm/job-run.sh qe-gpu python3 scripts/26_qe_smoke.py --help
```

## Submitting jobs

After install, example batch scripts are in `env/slurm/examples/`:

```bash
# Confirmatory QE GPU labels on spark-5da5
sbatch env/slurm/examples/sbatch-qe-gpu-labels.sh \
  --root runs/autonomous_electrodefect_20260607_041730/geometry_repair_convergence_gate \
  --max-seconds 7200

# KPM fraction sweep on spark-0808
sbatch env/slurm/examples/sbatch-kpm-sweep.sh \
  --morphology random --frac 0.35 --label slurm_smoke

# BigDFT CPU label (either node, no GPU reservation)
sbatch env/slurm/examples/sbatch-bigdft-labels.sh \
  --root runs/autonomous_electrodefect_20260607_041730/geometry_repair_convergence_gate \
  --max-jobs 1
```

Monitor:

```bash
sinfo -Nel
squeue -u "$USER"
tail -f /var/log/slurm/flash-modelling/job-<jobid>.log
```

## Integration with existing Python queues

The manifest-driven scripts (`27_phase2_qe_gpu_labels.py`, `14_large_lattice_kpm_showcase.py`, etc.) do **not** need to be rewritten immediately. The first integration step is:

1. Submit the whole queue script via `sbatch` with the right profile and `--nodelist`
2. Keep `status.json` / `heartbeat.log` semantics unchanged inside the script
3. Let Slurm enforce GPU exclusivity and timeouts

Later, individual jobs from a manifest can be split into Slurm job arrays if needed.

## Notes

- **QE GPU** (`qe-gpu` profile) expects NVHPC + QE 7.5 under `/home/ricfulop/Desktop/Cursor/.local-qe` on spark-5da5 only.
- **Peer repo** must be present at `/home/nvidia/Cursor/flash-modelling` (git pull after push).
- **Slinky/Kubernetes** is not part of this setup; bare Slurm is the right layer for two DGX nodes.
- Slurm job logs also go to `runs/slurm-<name>-<jobid>.{out,err}` via the example `#SBATCH` directives.

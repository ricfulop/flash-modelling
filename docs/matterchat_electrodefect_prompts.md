# MatterChat Electrodefect Prompt Pack

Use these prompts only after a candidate structure has passed geometry QC. MatterChat responses are qualitative brainstorming notes; they are not evidence for emission, work function, transport, barriers, or hot-electron energies.

## Lessons from the plan-review pass

The first plan-review attempt showed that MatterChat is sensitive to prompt length and may return empty, generic, or incoherent output when asked to review a broad analysis plan. Use it in short structure-grounded turns, and reject responses that fail the quality gate.

Observed outcomes from `runs/matterchat_geometry_qc_20260607_022909/matterchat_plan_review/`:

- Long plan-review prompts failed with generation-length errors.
- Short plan-review prompts completed but returned empty strings.
- Native structure prompts returned generic or incoherent text, including an unsupported `Nb` comparison.

Quality gate for accepting a MatterChat suggestion:

- response state is `completed`;
- output is non-empty;
- output is specific to the supplied material or structure;
- output contains no obvious material hallucination;
- output maps to a geometry, MLIP, DFT, transport, or TDDFT check.

If the response fails this gate, log it and do not use it to modify confirmatory modelling choices.

## Worker workflow

Start the CUDA worker from a normal terminal:

```bash
docker run --rm -it --gpus all --name matterchat-cuda \
  -v /home/ricfulop/Desktop/Cursor/matterchat:/workspace/matterchat \
  -v /home/ricfulop/Desktop/Cursor/flash-modelling:/workspace/flash-modelling:ro \
  -w /workspace/matterchat/src/MatterChat_code \
  nvcr.io/nvidia/pytorch:25.11-py3 bash
```

Inside the container:

```bash
python matterchat_worker.py --config ./config/inference_MatterChat.yaml --device cuda:0
```

Submit a QC-passing CIF from the host:

```bash
cd /home/ricfulop/Desktop/Cursor/matterchat
mkdir -p runtime_input_cifs
cp /home/ricfulop/Desktop/Cursor/flash-modelling/runs/<qc-run>/matterchat_inputs/<candidate>.cif \
  runtime_input_cifs/<candidate>.cif
python submit_matterchat_query.py \
  "$(/usr/bin/python3 -c 'from pathlib import Path; print(Path("/home/ricfulop/Desktop/Cursor/flash-modelling/docs/matterchat_prompt_current.txt").read_text())')" \
  --cif-path /workspace/matterchat/runtime_input_cifs/<candidate>.cif \
  --max-length 256 \
  --num-beams 3
```

If you start a fresh container with `/workspace/flash-modelling` mounted, direct `/workspace/flash-modelling/...` CIF paths are also valid. For an already-running worker, verify the path is visible before submitting.

## System boundary to include in every prompt

```text
You are being used for qualitative structure-property brainstorming only. Do not claim to prove electrodefect emission. Do not estimate emission rates, work functions, hot-electron energies, current thresholds, or transport coefficients. Identify local structural motifs or coordination changes that a separate DFT/transport workflow should test.
```

## Prompt 1: Local defect motifs

```text
You are being used for qualitative structure-property brainstorming only. Do not claim to prove electrodefect emission. Do not estimate emission rates, work functions, hot-electron energies, current thresholds, or transport coefficients. Identify local structural motifs or coordination changes that a separate DFT/transport workflow should test.

For this W/Mo BCC slab with a near-surface Frenkel-pair defect configuration, describe the local coordination motifs that look most relevant to defect recombination. Focus on vacancy-interstitial proximity, under-coordinated surface atoms, crowdion-like alignments, and any motifs that might plausibly change local electronic localization. Return a short list of testable motif hypotheses.
```

## Prompt 2: Surface and escape-relevant motifs

```text
You are being used for qualitative structure-property brainstorming only. Do not claim to prove electrodefect emission. Do not estimate emission rates, work functions, hot-electron energies, current thresholds, or transport coefficients. Identify local structural motifs or coordination changes that a separate DFT/transport workflow should test.

For this geometry-QC-passing defect slab, identify qualitative structural features near the emitting surface that might affect surface dipoles, local work-function variation, or vacuum-coupled electronic states. Do not compute any values. Phrase each suggestion as a DFT-testable question.
```

## Prompt 3: Localization-relevant disorder motifs

```text
You are being used for qualitative structure-property brainstorming only. Do not claim to prove electrodefect emission. Do not estimate emission rates, work functions, hot-electron energies, current thresholds, or transport coefficients. Identify local structural motifs or coordination changes that a separate DFT/transport workflow should test.

For this defect-network slab, qualitatively compare whether the visible defect arrangement looks more like an ordered network, random disorder, or a dendritic/percolating cluster. What local motifs might influence localization or hopping pathways? Return only qualitative observations that can be checked later with KPM or Wannier/downfolded Hamiltonians.
```

## Prompt 4: Negative-control prompt

```text
You are being used for qualitative structure-property brainstorming only. Do not claim to prove electrodefect emission. Do not estimate emission rates, work functions, hot-electron energies, current thresholds, or transport coefficients. Identify local structural motifs or coordination changes that a separate DFT/transport workflow should test.

This is a clean or recombined control slab. Identify which defect-related motifs are absent compared with a separated Frenkel-pair structure, and list what should not be interpreted as electrodefect-specific. Return possible control checks, not positive claims.
```

## Response logging template

Record every MatterChat response in a small JSON or markdown note:

```text
structure_path:
prompt_id:
prompt_text:
matterchat_response:
accepted_motif_suggestions:
rejected_or_unusable_suggestions:
physics_test_needed:
notes_on_hallucination_or_overclaiming:
```

## Promotion rule

A MatterChat suggestion can only promote a modelling variant if it is translated into:

- a named structure or motif;
- a geometry QC criterion;
- a DFT or transport observable;
- a fixed success/failure gate;
- and a statement that the MatterChat text itself is not evidence.

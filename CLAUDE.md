# Rules for AI-assisted sessions in this folder

This folder is shared with other team members. Sessions lose their context, so everything needed to continue lives in files.

## At the start of every session

Read docs/MAP.md, STATUS.md and README.md; docs/ARGUMENT.md and docs/GAP.md for the reasoning. SPEC.md is the binding design; read the parts you need.

## Keeping the record

- Update STATUS.md in the same turn as any change: jobs started or finished (with command, log file and next step), results, decisions with their reasons.
- Design changes go into SPEC.md and get a dated line under Decisions in STATUS.md; changes of reasoning go into docs/ARGUMENT.md.
- Commit your own files only (`git add <paths>`, never `git add -A` while others have uncommitted work) and push after each step. Pull first; push only as a fast-forward.

## Division of work (Wesley, 17 Oct 2026)

- Claude decides what is tested and how, writes the specification and briefs, and reviews every agent output before it is used.
- Code is written by GPT-6 Luna agents (Codex CLI, reasoning effort xhigh) through the user-level `codex-agents` skill. Briefs go in `tasks/`, answers in `out/`.
- Codex agents can only run the Codex Python runtime (numpy and pandas; no torch, no matplotlib). They write and syntax-check torch and plotting code; Claude runs it with the user's Python.

## Compute

- Wesley's laptop: Intel i5-8365U, 4 physical cores (8 threads), 16 GB RAM, no CUDA GPU. PyTorch CPU build.
- Team GPU (teammate, 23 Oct 2026): one NVIDIA GTX 1080 Ti on "snowflake". No access to ETH's Euler cluster; do not claim it.
- Quick runs by default (Wesley, 17 Oct 2026): a test of an idea should run in minutes (one or two settings, one seed, small n). Run a large grid only when that is clearly what is wanted. State the expected runtime before launching; above about 15 minutes, start with the quick version and offer the larger one.
- No usable GPU: the Intel UHD 620 has no CUDA, and DirectML on it would be slower than the CPU for these small sequential models.
- One compute job at a time; `--workers 4` with one thread per fit. Run long jobs in the background and record them under "Running now".

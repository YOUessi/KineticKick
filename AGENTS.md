# KineticKick working agreement

## Research goal and canonical roadmap

- The canonical, forward-looking objective and module plan is [docs/RESEARCH_OBJECTIVE_AND_ROADMAP.md](docs/RESEARCH_OBJECTIVE_AND_ROADMAP.md); use it before planning new scientific work. The executable backlog is [docs/NEXT_IMPLEMENTATION_BACKLOG.md](docs/NEXT_IMPLEMENTATION_BACKLOG.md), and experimental/method boundaries are in [docs/KINETICMIMIC_METHOD_AND_EVALUATION.md](docs/KINETICMIMIC_METHOD_AND_EVALUATION.md).
- The research goal is humanlike, novel, natural, whole-body high-impact instep soccer actions learned from qualified human motions; accuracy, legal foot/ball impact, full 29DoF safety, support, collision and recovery are hard gates. Old PAiD templates, tiny retarget residuals, joint MSE, offline QP, and old v32 successes are never sufficient to certify a new learned kick.
- Prioritize **qualified human football data and full-body motion tracking**, then controlled latent/contact-condition learning and genuine new-model G1 PhysX, rather than repeatedly increasing version numbers with CPU-only synthetic trajectories. Preserve failures, legitimate source licenses, clean train/validation/test separation and all source hashes.
- Existing [docs/RESEARCH_BATCH_SERIES_2026-11-11.md](docs/RESEARCH_BATCH_SERIES_2026-11-11.md) is an archived historical batch plan, NOT the current goal/priority source. This directive does not authorize terminating another agent's long-running GPU jobs.


## Version control

- The repository is https://github.com/YOUessi/KineticKick.
- Every new version must have its own branch, named `version/v<major>.<minor>.<patch>`.
- Create the version branch before making changes for that version. Continue work on the same branch only while it belongs to that version.
- Use the version number requested by the user. If none is specified, inspect existing local and remote version branches and choose the next unused patch version.
- Start from the appropriate verified version or `main`; keep `main` as the integration baseline. Do not implement a new version directly on `main`.
- Preserve previous version branches. Do not overwrite their history or force-push.
- Push a version branch with its upstream tracking reference when publishing that version.
- The initial import is `version/v0.1.0`; see `VERSION_CONTROL.md` for the workflow.

## Local research assets

- Preserve the existing Tang data, checkpoints, experiments, and simulator installation.
- Honor `.gitignore`: raw motion data, model weights, generated outputs, media, and upstream asset copies stay local.
- Keep upstream repository revisions and setup notes in `VERSION_CONTROL.md`.
- Run checks appropriate to the change and report the actual results. The complete test suite requires Tang-local assets; do not present a fresh clone as self-contained.

# KineticKick working agreement

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

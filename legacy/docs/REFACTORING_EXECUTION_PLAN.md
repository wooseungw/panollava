# Refactoring Execution Plan (Options 1–2)

This plan covers the agreed scope: hygiene plus module-level refactors for configuration, resampler registry, and script unification. Follow these steps incrementally to avoid regressions.

## 1) Hygiene (completed in this pass)
- Added `.pre-commit-config.yaml` with ruff/black/isort/mypy and basic sanity hooks.
- Added `Makefile` targets: `install`, `install-dev`, `lint`, `format`, `typecheck`, `test`, `hooks`, `clean`.
- Expanded `.gitignore` to cover pyc caches, build/dist, checkpoints, artifacts, logs.

## 2) Configuration system consolidation
- Source of truth: `src/panovlm/config/schema.py` and YAML configs in `configs/`.
- Actions:
  - Introduce a single loader entrypoint (e.g., `panovlm.config.loader.load_config(path: str) -> ModelConfig`) used by all scripts.
  - Deprecate `config_legacy.py*`; keep a thin compatibility shim that warns when used.
  - Validate configs early: schema validation + required stage fields; surface friendly errors.
  - Centralize default values in schema, not scattered across scripts.

## 3) Resampler registry
- Goal: one registry for all resampler implementations.
- Actions:
  - Create `src/panovlm/models/resampler/registry.py` with a mapping `{name: cls}` for `mlp`, `qformer`, `perceiver`, `bimamba`, `spatial_pool`, etc.
  - Provide `build_resampler(name: str, **kwargs)` that raises a clear error on unknown names and keeps a list of supported keys.
  - Update model assembly (`models/model.py` or factory) to call the registry instead of ad-hoc if/else.
  - Document supported names and expected kwargs in `docs/RESAMPLER_USAGE_GUIDE.md`.

## 4) Script unification
- Goal: reduce N entrypoints to one stable CLI per concern.
- Actions:
  - Keep: `scripts/train.py`, `scripts/eval.py`, `scripts/simple_inference.py` as primary CLIs.
  - Move specialized utilities (viz, cost profiling, dataset validation) under `tools/` or `scripts/experimental/` and mark them in docs.
  - Create shared argument helpers (e.g., `scripts/cli_utils.py`) for config loading, logging, stage selection to avoid drift.
  - Update `scripts/SCRIPTS_USAGE.md` to reflect the canonical set and deprecate others with pointers.

## 5) Safety checklist before merging refactors
- Run `make lint`, `make typecheck`, `make test` locally (or the subset available).
- Ensure configs in `configs/` load via the unified loader and that stage selection works.
- Verify at least one training/eval dry-run uses the registry path (unit smoke test).
- Update README references to scripts/config entrypoints.

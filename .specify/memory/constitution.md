<!--
Sync Impact Report
- Version change: 1.0.0 → 1.0.1
- Modified principles: Testing & Environments Discipline (manual test naming + artifact location clarifications)
- Added sections: None
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md → ✅ updated
  - .specify/templates/spec-template.md → ✅ updated
  - .specify/templates/tasks-template.md → ✅ updated
  - .specify/templates/checklist-template.md → ✅ updated
  - .specify/templates/commands/*.md → N/A (directory not present)
- Follow-up TODOs: None
-->

# LLM Performance Optimization Constitution

## Core Principles

### Pythonic Clarity & Readability (NON-NEGOTIABLE)

- Code MUST be clear, readable, and approachable for Python developers.
- All public functions, classes, and modules MUST include NumPy-style docstrings.
- Documentation and examples MUST prefer concrete Python code blocks that illustrate
  intended interfaces and behavior.
- Prefer reusable, modular components with separation of concerns; clarity and
  maintainability take precedence over cleverness.

Rationale: Improves onboarding speed, reliability, and shared understanding for a
Python-first contributor base.

### Typed, Linted, and Formatted (GATED)

- All parameters and return values MUST be type-annotated.
- Code MUST pass `mypy` type checks with zero errors.
- Code MUST pass `ruff` linting and formatting with zero errors.
- Prefer absolute imports; group imports by stdlib, third-party, and local modules.

Rationale: Strong typing and consistent style prevent defects and enable safe
refactoring in a performance-sensitive codebase.

### Object‑Oriented Design Discipline (Functional Classes)

- Follow a strict OO style for functional/service classes.
- Prefix member variables with `m_` and initialize them in `__init__`
  (default to `None` where appropriate).
- Provide read-only access via `@property`; perform mutations via explicit
  `set_xxx()` methods with validation.
- Keep constructors argument-free; provide factories as `@classmethod` factories
  `from_xxx()` for initialization.

Rationale: Enforces predictable state management and encapsulation for
performance-critical components.

### Data Models via attrs/pydantic (Data Classes)

- Use framework-native field naming (no `m_` prefix) for data models.
- Default choice: use `attrs` for most data models.
- Use `pydantic` for models intended for web request/response schemas and
  validation.
- Define fields declaratively with types and defaults; avoid business logic in
  data models.
- Prefer `@define(kw_only=True)` and `field()` for `attrs` with helpful
  `metadata` (e.g., `{ "help": "..." }`).

Rationale: Clear, declarative models improve validation, serialization, and
schema generation without coupling business logic into data containers.

### Testing & Environments Discipline

- Avoid system Python; prefer managed environments in priority order:
  1) Pixi environment (preferred), 2) Python virtualenv, 3) System Python (last
  resort).
- Documentation and scripts MUST state the expected runtime environment
  (e.g., `pixi run`, `pixi shell`, or activated venv).
- For major functionality, provide manual test scripts that a human can run and
  inspect; emphasize visualization/inspectable outputs.
- Manual tests MUST live under `tests/manual/` and MUST be prefixed with
  `manual_*.py` to avoid pytest collection.
- If automated tests are requested or warranted, use `pytest` for unit and
  integration tests with clear directory conventions:
  - Unit: `tests/unit/<subdir>/test_<name>.py`
  - Integration: `tests/integration/<subdir>/test_<name>.py`
  - Manual: `tests/manual/<feature_area>/manual_<name>.py`

Rationale: Clear environments and pragmatic testing strategies ensure reliable
reproducibility while maximizing value where automation is not yet essential.

## Performance & Profiling Standards

- Performance-sensitive changes SHOULD include profiling evidence when practical.
  Rationale: balances developer effort against the value of empirical evidence in
  a performance-focused project.
- Use NVIDIA Nsight Systems (`nsys`) to capture GPU/CPU timelines; include NVTX
  annotations where applicable for readable traces.
- Prefer installation via Pixi/conda‑forge when possible; document exact capture
  command lines and outputs under `tmp/`.
- Profiling/evaluation runs SHOULD write artifacts under
  `tmp/profile-output/<run_id>/` (or a subdirectory under `tmp/`) for discoverable,
  non-invasive results.
- Profiling artifacts MUST be attributable to the code change (document model,
  data, and hardware context briefly in notes).

## Development Workflow & Quality Gates

- All PRs MUST pass `mypy` and `ruff` checks with zero errors.
- All public APIs/classes MUST have NumPy-style docstrings and examples.
- Functional/service classes MUST follow the OO discipline in this constitution
  (member variable prefixing, properties, setters, factories).
- Data models MUST use `attrs` (default) or `pydantic` (for web schemas) with
  declarative fields and no embedded business logic.
- Feature work MUST declare its runtime environment context (Pixi preferred) and
  provide manual tests for major functionality; automated tests when requested
  or for critical code paths.
- Code review MUST explicitly confirm compliance with all Core Principles.

## Governance

- This constitution supersedes ad‑hoc practices; conflicts are resolved in favor
  of this document.
- Amendments are proposed via PRs that include:
  - Summary of the change and rationale
  - Version bump per semantic versioning rules (below)
  - Migration/impact notes and required template updates
- Ratification requires approval from at least one maintainer; emergency fixes
  may merge with post‑hoc notification.
- Versioning policy (semantic):
  - MAJOR: Backward‑incompatible governance or principle removals/redefinitions
  - MINOR: New principle/section added or materially expanded guidance
  - PATCH: Clarifications, wording, typo fixes, non‑semantic refinements
- Compliance review: PR reviewers MUST verify constitution gates are satisfied;
  CI SHOULD enforce `mypy`/`ruff` and any declared test steps.

**Version**: 1.0.1 | **Ratified**: 2025-10-28 | **Last Amended**: 2026-01-16

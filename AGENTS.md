# Project

Winnow (`winnow-fdr` on PyPI) is a library and CLI tool for confidence calibration and FDR control for *de novo* peptide sequencing. The installable package is `winnow/`. Nothing else at the repo root is published API. This is an open-source repo, so root files that look internal are still public.

| Path | Purpose | When to change |
|------|---------|----------------|
| `scripts/` | General research and one-off helpers | Freely; never import into `winnow/` |
| `paper_scripts/` | Reproduces the associated paper only | Only to keep it working with changes in `winnow/`; no features or refactors |
| `Makefile.paper` | CLI commands for paper reproduction; pairs with `paper_scripts/` | Only when asked, or to stay in sync with `paper_scripts/` and the pinned artefact versions |
| `Dockerfile`, `.dockerignore`, `manifest.yaml` | Scaffolding for launching experiments on InstaDeep's internal cluster | Only on an experiment branch and when expressly asked. CI (`.github/workflows/manifest.yml`) blocks PRs to `main` unless `manifest.yaml` is the toy default |
| `Makefile` | Two roles: (1) reusable commands for QC and running the examples, for users and contributors; (2) experiment commands, written as Make targets with self-explanatory names and called from the `manifest.yaml` `command` | (1) Normal development. (2) Only on experiment branches or while testing a fix or feature, and when asked |
| `assets/` | Images, GIFs and videos used by the docs | When docs need media |
| `examples/` | Example notebooks and their toy data | Keep real data cut to a short file size; do not generate synthetic dummy data where real data can be trimmed |

Layout, conventions, CI and the release process: `docs/contributing.md`.

# Workflow

- **Deps**: `uv` (`uv.lock` committed). Use `uv add` / `uv sync`; never hand-edit `uv.lock`; no `pip`, `poetry` or `conda`.
- **torch**: the `torch` pin in `pyproject.toml` must stay compatible with InstaNovo's, so both install in one environment for end-to-end pipelines. Check InstaNovo's torch constraint before touching it.
- **Env**: check for `.venv/` before creating one; otherwise `uv venv && uv sync`. Run Python via `uv run`.
- **Tasks**: `make` is the entry point (`make install-dev`, `make tests`, `make train-sample`, `make predict-sample`).
- **Verify**: `make tests` and `pre-commit run --all-files`.
- **Commits**: Conventional Commits (pre-commit enforces). `exp` is only for launching cluster experiments; never for core code changes, and never merged to `main`.
- **Attribution**: do not author commits, issues or PRs as an AI agent: no AI `Co-Authored-By` trailer, no "Generated with" line. Trailers crediting human authors are fine.
- **Tagging**: never @-mention, tag or request a review from anyone in issues, PRs, descriptions, comments or commit messages unless the user explicitly names who.
- **Branches**: hyphenated prefix, never a slash: `feat-amazing-feature`, `fix-calibration-bug`.
- **Issues / PRs**: pick the matching `.github/ISSUE_TEMPLATE/` form. Read `.github/pull_request_template.md` before writing a PR body and fill every section; `gh pr create --body` bypasses templates, so paste its headings.

# Writing: commits, issues, PRs, comments, docstrings, docs, replies

Prose in a PR description gets the same scrutiny as prose in a docstring.

- Answer first. Reasoning only where it changes a decision. No preamble, no recap.
- Cut every sentence that carries no information. Prefer a line to a paragraph, a table to prose.
- Concrete over abstract: numbers, paths, symbols. Not "robust", "seamless", "significantly improved".
- If the diff mixes unrelated changes, suggest splitting the PR instead of covering them in one description.
- Describe what the change does and how it was verified. Do not narrate the debugging story, and do not bring in context from downstream projects.
- No slop: binary contrasts ("not X, but Y"), throat-clearing, dramatic fragments, personification, puffery, fake-profound endings.
- No emojis, no em dashes, sparse bolding. English spelling (colour, organise, behaviour). No Oxford commas unless needed.
- Be critical of design when a better approach exists; no sycophancy.
- Credit prior work when reimplementing something that exists on another branch: a note in the PR description, or a `Co-Authored-By` trailer naming the human author.

# Documentation

`docs/` and `docs/contributing.md` are written for human readers first. Agent instructions live in this file (`AGENTS.md`), never in the docs.

One page per purpose, one home per fact. The reasoning and the table of which page owns what are in the Documentation section of `docs/contributing.md`; read it before writing docs.

- Find the page that owns the topic and edit it. Do not create a new page or copy content from another.
- Do not restate docs content in the README, PR descriptions or code comments. Link to it.
- Code blocks in docs must run as written on the example data.
- A change that alters behaviour, CLI or config updates the owning page in the same PR.
- A page opens with one sentence saying what it covers.
- `docs/api/` is hand-written; update it when a public class or function changes.
- Do not reorganise existing pages unasked.

# Minimum viable diff

- Every changed line traces to the request. No unrequested refactor, rename, reformat or "while I'm here" cleanup.
- If nearby code looks outdated but is not needed for the task, leave it and say so.
- Match surrounding style, naming and patterns; do not introduce a new one.
- **Duplicated logic**: when the same logic (not just similar-looking code) is about to exist in a second place in `winnow/`, extract it into one helper that both import, for example the Koina client construction repeated across the Koina features. Put it low enough in the import graph to avoid cycles (`winnow/utils/` or the nearest shared module).
- **New abstraction**: do not add a base class, factory, wrapper or config indirection for fewer than 3 real callers or implementations. Extend the existing interfaces (`DatasetLoader`, `CalibrationFeatures`, `FeatureDependency`, `FDRControl`) instead of adding layers.
- Rule of thumb: a helper at the second copy, an abstraction at the third caller.
- No caching, batching or extra logging without a named reason. Log at decision points and real boundaries only.
- Put behaviour where its purpose lives (e.g. I/L equivalence inside the matching function, not in each caller) so callers cannot forget it.

# Docstrings

Google style, enforced by pydocstyle. State why, not what: the name and signature already say what.

- **Public functions, methods and classes**: a one-line summary, then Google `Args`, `Returns` and `Raises`. Add side effects and restrictions on when it can be called where they apply. Mark optional arguments and say whether keyword arguments are part of the interface.
- **Private functions and tests**: one sentence.
- **Multi-line prose** is the exception, for complex logic or complicated use cases. It must make sense to a reader with no knowledge of how the code came to be: no "now", "previously", PR numbers or context from the session that wrote it.
- **Classes**: a blank line between the class docstring and the first method. Document the constructor on `__init__`. A subclass docstring says what it inherits and summarises the differences, using "override" when a method replaces the parent's without calling it and "extend" when it calls the parent.
- **Standalone scripts**: the module docstring doubles as the usage message: what the script does, command-line syntax, environment variables and files.
- **Library modules and packages**: a docstring is optional. Add one only when the purpose is not clear from the name and contents. When present it lists the public classes and functions (for a package, its modules and subpackages) with one-line summaries.

# Code style (not fully caught by linters)

- Type every new or changed function. New annotations use `X | None`, `list[T]`, `dict[K, V]`, not `Optional` / `List` / `Dict`. Leave untouched older annotations alone.
- Comments: only for a non-obvious *why*. No narration of the edit, no history, no references to PRs or downstream tools. Keep existing comments and `# TODO` / `# FIXME` unless wrong.
- f-strings; `pathlib.Path` over `os.path`; global imports unless circular or slow at CLI startup.

# Non-negotiables

- **Package boundary**: public behaviour belongs in `winnow/`. Never import from root `scripts/` or `paper_scripts/` into `winnow/`.
- **Logging**: `logger = logging.getLogger(__name__)`. No `print` under `winnow/` (the CLI may use Rich).
- **Datasets**: load through a `DatasetLoader` (`winnow.datasets.interfaces`) into `CalibrationDataset`. Use the existing loaders rather than ad-hoc parsing.
- **Features**: subclass `CalibrationFeatures` / `FeatureDependency` in `winnow.calibration.features` and wire through Hydra configs.
- **FDR**: extend `FDRControl` in `winnow.fdr`; keep database-grounded and nonparametric paths explicit.
- **Calibrators**: load with `ProbabilityCalibrator` (local directory or Hugging Face id). Do not hand-roll checkpoint loading.
- **Checkpoints**: do not persist runtime-only settings (Koina server, collision energy, etc.). Add them to `KOINA_RUNTIME_CONFIG_KEYS` so save/load strips them.
- **Config**: Hydra YAML and `hydra.utils.instantiate`. Resolve config dirs through `winnow.utils.config_path`.
- **Shared types**: reuse `winnow.data_types` (`Peptide`, `Spectrum`); do not redefine aliases.

# Quality

- No dead code. Grep tests, configs, CLI entry points, notebooks and docs before deleting.
- No circular imports. Extract shared pieces downward; avoid `TYPE_CHECKING` and local-import workarounds.
- No weak types: `Any`, bare `dict` / `list`, `object` or blanket `# type: ignore` need a one-line justification.
- No defensive `try/except`. Catch only at real boundaries (user input, IO, network, subprocess) with a concrete handling strategy. Never `except Exception: pass`.
- Backwards compatibility is a per-task decision. If a change could alter outputs for existing inputs or break saved checkpoints (e.g. I/L handling), say so and ask how to handle it before implementing. Do not add or remove compatibility code unprompted.
- Otherwise one code path. No speculative "just in case" branches, and no compatibility code without a named consumer (`winnow/compat/` included).

# Testing

- Mirror the source layout under `tests/`. Files `test_*.py`, functions `test_*`.
- When code moves or is renamed, move its tests verbatim alongside it and update only import paths and names. Grep `tests/` and call sites for the old path.
- Name tests for what they check, with a one-sentence docstring at most (see Docstrings). Test the unit in its own module's test file.
- `@pytest.mark.parametrize` over loops; `pytest.raises(SpecificError)`, never bare `Exception`.
- Reuse fixtures from existing `conftest.py` files.
- Do not use TDD unless asked.

# Secrets

- Never commit secrets, service-account JSONs, tokens or DB URLs. `ext-*gcp*.json` and `*-creds*` are sensitive.
- Read env vars at entry points and pass values down; no `os.environ` deep in `winnow/`.

# Ask before

- Changing dependency versions in `pyproject.toml` / `uv.lock`, or adding a dependency.
- Deleting files (source, config, docs, notebooks, data).
- Deleting an existing test or changing what it asserts. A verbatim move with updated paths and names is fine, as are new tests.
- Pushing, force-pushing, or editing CI workflows / Docker release wiring.
- Committing large binaries (`.safetensors`, bulky parquet, raw spectra).

A one-off "ok" covers that session and operation only.

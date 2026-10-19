# Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire and create, and we welcome your support. Any contributions you make are greatly appreciated.

## Ways to contribute

If you have ideas for enhancements, you can:

- Fork the repository and submit a pull request
- Open an issue using the form that matches your topic (feature, bug, docs, and so on)
- Improve documentation
- Add examples or tutorials

## Contribution process

1. **Fork the repository**

    ```bash
    git clone https://github.com/instadeepai/winnow.git
    cd winnow
    ```

2. **Create a feature branch**

    ```bash
    git checkout -b feat-amazing-feature
    ```

3. **Set up development environment**

    ```bash
    # Install the project with development tools (uv)
    make install-dev

    # Set up pre-commit hooks
    pre-commit install
    ```

4. **Make your changes**

    - Follow the existing code style
    - Add tests for new functionality
    - Update documentation as needed

5. **Test your changes**

    ```bash
    # Run tests
    make tests

    # Check code formatting
    pre-commit run --all-files
    ```

6. **Commit your changes**

    ```bash
    git commit -m 'feat: add some amazing feature'
    ```

7. **Push to your branch**

    ```bash
    git push origin feat-amazing-feature
    ```

8. **Open a Pull Request**

    The pull request template lists what to fill in. Reference any related issues and include examples if applicable.

## Conventions

These are enforced by pre-commit or CI where possible.

### Branches and commits

- **Branch names** use a hyphenated type prefix, not a slash: `feat-amazing-feature`, `fix-calibration-bug`.
- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/). The allowed types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, `perf` and `style`.
- **`exp` commits** are reserved for launching experiments on InstaDeep's internal compute cluster. They are never merged into `main`.

### Issues and pull requests

- Issues use the forms in `.github/ISSUE_TEMPLATE/`, one per commit type (feature, bug, refactor, test, docs, CI, build).
- Pull requests use `.github/pull_request_template.md`.

### Code and tests

- **Dependencies** are managed with `uv`; `uv.lock` is committed and should only change through `uv add`, `uv remove` or `uv lock`.
- The `torch` version range in `pyproject.toml` must stay compatible with [InstaNovo](https://github.com/instadeepai/InstaNovo), so that both install in one environment for end-to-end pipelines.
- **Tests** live under `tests/`, mirroring the layout of `winnow/`. Files are named `test_*.py` and run with `make tests`.
- **Linting and formatting** are configured in `.pre-commit-config.yaml`; `pre-commit run --all-files` runs them. CI runs the same checks, the tests across operating systems and Python versions, and the docs build. See `.github/workflows/`.

## Repository layout

The installable package is `winnow/`. The other folders support development, documentation, the paper and experiments. They are not part of the published API.

### The `winnow` package

| Path | Summary |
| ---- | ------- |
| **`winnow/datasets/`** | `CalibrationDataset`, `FeatureDataset` and PSM types, the `DatasetLoader` interface, and the loaders for each tool's output (`data_loaders/`: InstaNovo, MZTab, PointNovo, PrimeNovo, Winnow-native). |
| **`winnow/calibration/`** | `ProbabilityCalibrator`, the `CalibrationFeatures` base and the rescoring features (`features/`), and calibration diagnostics. |
| **`winnow/fdr/`** | Abstract `FDRControl`, then `DatabaseGroundedFDRControl` and `NonParametricFDRControl`. |
| **`winnow/scripts/`** | The Typer CLI behind the `winnow` console command. |
| **`winnow/configs/`** | Default Hydra YAML shipped with the package: one file per command, plus data loader and FDR method options. |
| **`winnow/utils/`** | Config path and override helpers, Koina and iRT configuration, peptide and proteome helpers, console output. |
| **`winnow/compat/`** | Compatibility shims (currently for InstaNovo). |
| **`winnow/data_types.py`** | Small shared type aliases (`Peptide`, `Spectrum`) used across modules. |

### Everything else

| Path | Summary |
| ---- | ------- |
| **`tests/`** | Unit tests, mirroring the layout of `winnow/`. |
| **`docs/`**, **`mkdocs.yml`** | The documentation site (MkDocs). |
| **`assets/`** | Images, GIFs and videos used by the docs. |
| **`examples/`** | Example notebook and associated data files. We prefer real data cut down to a short file size over synthetic dummy data. |
| **`scripts/`** | General one-off helpers. |
| **`paper_scripts/`**, **`Makefile.paper`** | Reproduce the figures and tables of the associated paper from public artefacts (see `paper_scripts/README.md`). |
| **`Makefile`** | Two roles. It holds commands that users and contributors reuse: install, tests, docs, building the package and running the examples. It also holds experiment commands, written as targets with self-explanatory names so the cluster `manifest.yaml` can call them. |
| **`Dockerfile`**, **`.dockerignore`**, **`manifest.yaml`** | Scaffolding for launching experiments on InstaDeep's internal compute cluster. The repository is open source, so they are visible, but they only matter on an experiment branch. On `main`, `manifest.yaml` stays a harmless default, and the Manifest check enforces this on pull requests. |
| **`.github/`** | CI workflows, issue forms, the pull request template and the pre-commit hook. |
| **`.pre-commit-config.yaml`**, **`setup_precommit.sh`**, **`.codespell-ignore.txt`** | Pre-commit hook configuration, a setup script that installs them, and words codespell should accept. |
| **`pyproject.toml`**, **`uv.lock`** | Package metadata and the locked dependencies. |
| **`CITATION.cff`**, **`LICENSE.md`**, **`README.md`** | Citation metadata, the licence and the project landing page. |

## Documentation

Good documentation depends on knowing which question each page answers. We keep each page to one purpose and each fact in one place for two reasons. Readers find an answer without comparing several overlapping pages, and a change only has to be made once.

If you change behaviour, the CLI or configuration, update the associated documentation page in the same pull request. If you find yourself copying a paragraph from another page, link to it instead.

| Purpose | Question it answers | Where it lives |
| ------- | ------------------- | -------------- |
| Landing | What is Winnow and how do I install it? | `README.md` (also the docs home) |
| Tutorial | Show me it working end to end | [Examples](examples.md) and the example notebook |
| Reference | What exactly does this command, key or function take and return? | [CLI](cli.md), [configuration](configuration.md) (alongside the comments in `winnow/configs/`), and `docs/api/` |
| Contributing | How do I change the code? | This page |

## Releases

A release publishes the package to PyPI. Archived files cannot be changed, so a tag cannot be reused. If a release goes wrong, release the next version number instead.

### Versioning

Releases are tagged `vX.Y.Z` and follow [semantic versioning](https://semver.org/):

- **Patch** (`2.1.0` to `2.1.1`): bug fixes and other changes that do not alter how existing code, configs or checkpoints behave.
- **Minor** (`2.0.0` to `2.1.0`): new, backwards-compatible functionality.
- **Major** (`1.x` to `2.0.0`): changes that can break existing use, such as different outputs for the same inputs, checkpoints that no longer load, changed CLI or config keys, or changes to the public API. Compatibility code for the previous major version is removed at a major release.

Release notes are the list of merged pull requests that GitHub generates when the release is created. There is no separate changelog file.

### Steps

1. **Bump the version in a pull request.** Update `version` in `pyproject.toml`, refresh `uv.lock` with `uv lock`, and update `version` and `date-released` in `CITATION.cff`. Merge it into `main`.
2. **Create a GitHub release from `main`** with a tag of the form `vX.Y.Z`, where `X.Y.Z` is the version from step 1.
3. **Check the publish workflow.** It reads the version from `pyproject.toml` and fails before building if the tag does not match it, or if that version is already on PyPI. Otherwise it builds the package and uploads it to PyPI using trusted publishing.
4. **Confirm the result** on [PyPI](https://pypi.org/project/winnow-fdr/).

## Getting help

If you need help or have questions:

- Check the [examples](examples.md) for usage patterns
- Look through existing [issues](https://github.com/instadeepai/winnow/issues)
- Open a new issue with the "question" label

## Recognition

Contributors are recognised in the project's contributor list. Don't forget to give the project a star! ⭐

Thank you for considering contributing to Winnow!

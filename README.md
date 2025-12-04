<a id="readme-top"></a>

<p align="center">
    <a href="https://www.python.org/downloads/release/python-390/">
        <img
            src="https://img.shields.io/badge/python-3.9+-blue.svg"
            alt="Python 3.9+"
            style="max-width:100%;"
        >
    </a>
    <a href="https://docs.astral.sh/ruff">
        <img
            src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"
            alt="Ruff"
            style="max-width:100%;"
        >
    </a>
    <a href="https://github.com/pre-commit/pre-commit">
        <img
            src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit"
            alt="pre-commit"
            style="max-width:100%;"
        >
    </a>
    <a href="https://github.com/instadeepai/winnow/actions">
        <img
            src="https://img.shields.io/badge/coverage-55%25-red?logo=coverage"
            alt="Test Coverage"
            style="max-width:100%;"
        >
    </a>
</p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h1 align="center">Winnow</h1>

  <p align="center">
    Confidence calibration and FDR control for <i>de novo</i> peptide sequencing
    <br />
    <a href="https://instadeepai.github.io/winnow/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/instadeepai/winnow/issues/new?labels=bug&template=bug_report.md">Report bug</a>
    &middot;
    <a href="https://github.com/instadeepai/winnow/issues/new?labels=enhancement&template=feature_request.md">Request feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>

<!-- WORKFLOW DIAGRAM -->
<div align="center">
  <img src="https://raw.githubusercontent.com/instadeepai/winnow/main/docs/assets/winnow_workflow.png" alt="Winnow Workflow" style="max-width:100%;">
  <p>Winnow workflow for confidence calibration and FDR control in <em>de novo</em> peptide sequencing</p>
</div>

<!-- ABOUT THE PROJECT -->
## About the project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
In bottom-up proteomics workflows, peptide sequencing—matching an MS2 spectrum to a peptide—is just the first step. The resulting peptide-spectrum matches (PSMs) often contain many incorrect identifications, which can negatively impact downstream tasks like protein assembly.

To mitigate this, intermediate steps are introduced to:

1. Assign confidence scores to PSMs that better correlate with correctness.
2. Estimate and control the false discovery rate (FDR) by filtering identifications based on confidence scores.

For database search-based peptide sequencing, PSM rescoring and target-decoy competition (TDC) are standard approaches, supported by an extensive ecosystem of tools. However, *de novo* peptide sequencing lacks standardised methods for these tasks.

`winnow` aims to fill this gap by implementing the calibrate-estimate framework for FDR estimation. Unlike TDC, this approach is directly applicable to *de novo* sequencing models. Additionally, its calibration step naturally incorporates common confidence rescoring workflows as part of FDR estimation.

`winnow` provides both a CLI and a Python package, offering flexibility in performing confidence calibration and FDR estimation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Installation

`winnow` is available as a Python package and can be installed using `pip` or a `pip`-compatible command (e.g., `uv pip install`):
```
pip install winnow-fdr
```
or
```
uv pip install winnow-fdr
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->
## Quick Start

Get started with `winnow` in minutes using the included sample data:

```bash
# Generate sample data (if not already present)
make sample-data

# Train a calibrator on the sample data
make train-sample

# Run prediction with the trained model
make predict-sample
```

The sample data is automatically configured in `config/train.yaml` and `config/predict.yaml`, so you can also use the commands directly:

```bash
# Train with sample data
winnow train

# Predict with pretrained HuggingFace model
winnow predict

# Predict with your locally trained model
winnow predict calibrator.pretrained_model_name_or_path=models/new_model
```

**Note:** The sample data is minimal (20 spectra) and intended for testing only. For production use, replace with your own datasets.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

`winnow` supports two usage modes:

1. A command-line interface (CLI) with sensible defaults and multiple FDR estimation methods.
2. A configurable and extensible Python package for advanced users.

### CLI

Installing `winnow` provides the `winnow` command with two sub-commands:

1. `winnow train` – Performs confidence calibration on a dataset of annotated PSMs, outputting the fitted model checkpoint.
2. `winnow predict` – Performs confidence calibration using a fitted model checkpoint (defaults to a pretrained general model from HuggingFace), estimates and controls FDR using the calibrated confidence scores.

By default, `winnow predict` uses a pretrained general model (`InstaDeepAI/winnow-general-model`) hosted on HuggingFace Hub, allowing you to get started immediately without training. You can also specify custom HuggingFace models or use locally trained models.

Winnow uses [Hydra](https://hydra.cc/) for flexible, hierarchical configuration management. All parameters can be configured via YAML files or overridden on the command line:

```bash
# Quick start with defaults
winnow predict

# Override specific parameters
winnow predict fdr_control.fdr_threshold=0.01

# Specify different data source and dataset paths
winnow predict data_loader=mztab dataset.spectrum_path_or_directory=data/spectra.parquet dataset.predictions_path=data/preds.mztab
```

Refer to the [CLI Guide](cli.md) and [Configuration Guide](configuration.md) for details on usage and configuration options.

### Package

The `winnow` package is organised into three sub-modules:

1. `winnow.datasets` – Handles data loading and saving, including the `CalibrationDataset` class for mapping peptide sequencing output formats.
2. `winnow.calibration` – Implements confidence calibration. Key components include:
    - `ProbabilityCalibrator` (defines the calibration model)
    - `CalibrationFeature` (an extensible interface for defining calibration features)
3. `winnow.fdr` – Implements FDR estimation methods:
    - `DatabaseGroundedFDRControl` (for database-grounded FDR control)
    - `NonParametricFDRControl` (uses a non-parametric and label-free method for FDR estimation)

For an example, check out the [example notebook](https://github.com/instadeepai/winnow/blob/main/examples/getting_started_with_winnow.ipynb).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire and create, and we welcome your support! Any contributions you make are **greatly appreciated**.

If you have ideas for enhancements, you can:

- Fork the repository and submit a pull request.
- Open an issue and tag it with "enhancement".

### Contribution process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to your branch (`git push origin feature/AmazingFeature`).

Don't forget to give the project a star! Thanks again! :star:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### BibTeX entry and citation information

If you use `winnow` in your research, please cite the following preprint:

```bibtex
@article{mabona2025novopeptidesequencingrescoring,
      title={De novo peptide sequencing rescoring and FDR estimation with Winnow},
      author={Amandla Mabona and Jemma Daniel and Henrik Servais Janssen Knudsen and Rachel Catzel
      and Kevin Michael Eloff and Erwin M. Schoof and Nicolas Lopez Carranza and Timothy P. Jenkins
      and Jeroen Van Goey and Konstantinos Kalogeropoulos},
      year={2025},
      eprint={2509.24952},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2509.24952},
}
```

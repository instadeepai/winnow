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
    <a href="https://github.com/instadeepai/winnow/issues/new?labels=bug&template=bug_report.md">Report Bug</a>
    &middot;
    <a href="https://github.com/instadeepai/winnow/issues/new?labels=enhancement&template=feature_request.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#CLI">CLI</a></li>
        <li><a href="#Package">Package</a></li>
      </ul>
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
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
In bottom-up proteomics workflows, peptide sequencing—matching an MS2 spectrum to a peptide—is just the first step. The resulting peptide-spectrum matches (PSMs) often contain many incorrect identifications, which can negatively impact downstream tasks like protein assembly.

To mitigate this, intermediate steps are introduced to:

1. Assign confidence scores to PSMs that better correlate with correctness.
2. Estimate and control the false discovery rate (FDR) by filtering identifications based on confidence scores.

For database search-based peptide sequencing, PSM rescoring and target-decoy competition (TDC) are standard approaches, supported by an extensive ecosystem of tools. However, *de novo* peptide sequencing lacks standardized methods for these tasks.

`winnow` aims to fill this gap by implementing the calibrate-estimate framework for FDR estimation. Unlike TDC, this approach is directly applicable to *de novo* sequencing models. Additionally, its calibration step naturally incorporates common confidence rescoring workflows as part of FDR estimation.

`winnow` provides both a CLI and a Python package, offering flexibility in performing confidence calibration and FDR estimation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Installation

`winnow` is available as a Python package and can be installed using `pip` or a `pip`-compatible command (e.g., `uv pip install`):
```
pip install winnow_fdr
```
or
```
uv pip install winnow_fdr
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

`winnow` supports two usage modes:

1. A command-line interface (CLI) with sensible defaults and multiple FDR estimation methods.
2. A configurable and extensible Python package for advanced users.

### CLI

Installing `winnow` provides the `winnow` command with two sub-commands:

1. `winnow train` – Performs confidence calibration on a dataset of annotated PSMs, outputting the fitted model checkpoint.
2. `winnow predict` – Performs confidence calibration using a fitted model checkpoint, estimates and controls FDR using the calibrated confidence scores.

Refer to the documentation for details on command-line arguments and usage examples.

### Package

The `winnow` package is organized into three sub-modules:

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

### Contribution Process

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to your branch (`git push origin feature/AmazingFeature`).

Don't forget to give the project a star! Thanks again! :star:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### BibTeX entry and citation info

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

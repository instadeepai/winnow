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
<h3 align="center">winnow</h3>

  <p align="center">
    Confidence calibration and FDR control for de novo peptide sequencing
    <br />
    <a href="https://github.com/instadeepai/winnow"><strong>Explore the docs »</strong></a>
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



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
Peptide sequencing - that is returning a peptide match for an MS2 spectrum - is typically only the first step in bottom-up proteomics workflows. The peptide-spectrum matches (PSMs) that result from this are typically noisy and contain many incorrect identifications. These will not usually be of high enough quality for error-sensitive tasks further along the pipeline such as protein assembly.

To address this problem, workflows have intermediate steps which try to sift the correct identifications from the incorrect ones. This falls into two sub-steps
1. assigning confidence scores to the PSMs which differ from those returned by the peptide sequencing systems and better correlate with correctness, and
2. estimating the false discovery rate (FDR) of identifications with confidence above a certain threshold and filtering according to a threshold which controls FDR to an acceptable level.

For peptide sequencing systems that use database search, PSM rescoring and target-decoy competition (TDC) are the standard approaches for performing these sub-steps and there is a wide ecosystem of tools that implement them. However, for de novo peptide sequencing, there are no standard approaches or tools.

`winnow` aims to step into this role. It implements the calibrate-estimate framework for FDR estimation. Unlike TDC, this framework is directly applicable to de novo sequencing models, and the calibrate step subsumes many common workflows for confidence rescoring as a natural part of FDR estimation.

`winnow` provides a CLI and a customizable and extensible `python` package for performing confidence calibration and FDR estimation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Installation

`winnow` is distributed as a `python` package. You can install both it and the CLI using `pip` or `pip`-equivalent commands (e.g. `uv pip install`).
```
pip install https://github.com/instadeepai/winnow
```
or
```
uv pip install https://github.com/instadeepai/winnow
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

`winnow` allows two modes of usage:
1. a command line interface with good default features and choice of FDR estimation method
2. an configurable and extensible `python` package

We describe usage of both below.

### CLI

Installing the package will also install the `winnow` command.
This has two sub-commands:
1. `winnow calibrate` for confidence calibration which takes as input a dataset of PSMs, and
2. `winnow estimate` for FDR estimation and filtering which takes as input a dataset of PSMs with calibrated confidence scores returned by `winnow calibrate`

You can consult the documentation for details about arguments and how to use these commands.

### Package

The `winnow` package is organised into three sub-modules:
1. `winnow.datasets` for data handling including loading and saving datasets. This contains the `CalibrationDataset` class which is used internally to map between different peptide sequencing output formats;
2. `winnow.calibration` for confidence calibration. The two key classes it contains are the `ProbabilityCalibrator` class - which defines the model used for confidence calibration - and the `CalibrationFeature` interface - which defines an extensible interface for features used for calibration. This sub-module also contains implementations of many features sub-classing the `CalibrationFeature` interface; and
3. `winnow.fdr` for FDR estimation. This contains the `DatabaseGroundedFDRControl` class which implements database-grounded FDR control, and the `EmpiricalBayesFDRControl` class which implements the empirical Bayesian mixture model for FDR calibration that the calibration-estimate framework is built on.

Please refer to the [example notebook](https://github.com/instadeepai/winnow/blob/main/examples/fdr_plots.ipynb) for an example of how to use the package.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

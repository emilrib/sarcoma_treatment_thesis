
These files centralise directory paths and modelling variables used throughout the pipeline.

## Model Outputs

The modelling pipeline saves trained objects and generated analysis outputs locally within the repository structure.

Generated files may include:

- Trained causal model objects
- Preprocessing objects
- Intermediate processed outputs
- Evaluation summaries
- Visualisation outputs

Generated files should be reviewed before sharing the repository externally.

## Data Confidentiality

This repository is designed for confidential research data.

Before sharing, publishing, or archiving the project, ensure that:

- No confidential data files are included.
- No identifiable information is present in committed files.
- Generated outputs are reviewed for sensitive information.
- Model artifacts are checked before distribution.
- Institutional, ethical, and legal data governance requirements are followed.

The `Datasets/` directory should be treated as confidential and should not be publicly distributed unless explicitly authorised.

## Validate Package

The `validate/` package contains local evaluation utilities for causal inference model assessment.

Some functionality is adapted from or inspired by evaluation tools in Microsoft Research's EconML library.

Original EconML repository:

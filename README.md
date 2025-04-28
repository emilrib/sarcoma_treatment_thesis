### Binary Codes Meaning
- `0`: **No**
- `1`: **Yes**

### Validate Package

This package provides local extensions to model evaluation tools for causal inference.

It adapts and extends functionalities originally developed in Microsoft Research's EconML library (v0.15.1), including:

- Uplift Ranking Evaluation(`UpliftEvaluationResults`)

#### Origin
The original methods are documented in the EconML repository: [https://github.com/microsoft/EconML](https://github.com/microsoft/EconML).

#### License
The original EconML library is under the MIT License. This project inherits that license.

### STUVA Analysis
The STUVA analysis uses ANOVA to determine if there is a significant STUVA interference amaong groups.
- `1`: If p-value is less than 0.05: Significant difference detected across groups — possible SUTVA interference.
- `2`: if p-value is higher than 0.05: No significant group-level differences detected — SUTVA may hold.

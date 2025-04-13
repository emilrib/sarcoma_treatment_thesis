### Code Legends
#### Gender
- `1`: **Female**
- `2`: **Male**

#### Metastasis Classification
- `1`: **Metastasis free** during treatment observation
- `2`: Metastasis existed **before** first patient contact
- `3`: Metastasis appeared **after** first patient contact
- `0`: Metastasis present, discovery date unknown

#### Patient Status
- `1`: **NED** (No evidence of disease)
- `2`: **AWD** (Alive with disease)
- `3`: **DOD** (Dead of disease)

#### Remaining Labels (Binary Codes)
- `1`: **No**
- `2`: **Yes**

### STUVA Analysis
The STUVA analysis uses ANOVA to determine if there is a significant STUVA interference amaong groups.
- `1`: If p-value is less than 0.05: Significant difference detected across groups — possible SUTVA interference.
- `2`: if p-value is higher than 0.05: No significant group-level differences detected — SUTVA may hold.
